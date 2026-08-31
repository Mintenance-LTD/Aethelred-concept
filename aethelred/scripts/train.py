"""
Train the Aethelred tactical AI with production pipeline.

Features:
  - PPO with GAE (Generalized Advantage Estimation)
  - Curriculum learning (progressive difficulty)
  - TensorBoard logging
  - Checkpointing (best + periodic)
  - Sensor noise injection for robustness
  - Integrated safety system validation
  - Adaptation engine (MAML + EWC) for online threat adaptation

Usage:
    python scripts/train.py --mode online --episodes 100
    python scripts/train.py --mode curriculum --episodes 200
    python scripts/train.py --mode stress --episodes 50
    python scripts/train.py --resume checkpoints/best_policy.pt
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.adaptation.opponent_model import OpponentBehaviorModel
from aethelred.config.settings import AethelredConfig
from aethelred.deployment.safety import SafetyConfig, SafetyExecutionGateway, SafetyManager
from aethelred.learning.learning_loop import LearningLoop
from aethelred.learning.trainer import CurriculumStage, PPOTrainer
from aethelred.simulation.environment import AethelredEnv
from aethelred.simulation.scenarios import apply_scenario_to_config, load_scenario
from aethelred.tactical_ai.policy import TacticalPolicy
from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder
from aethelred.utils.seeding import set_seed

ACTION_TYPE_NAMES = [
    "move", "engage", "evade", "formation_change",
    "recon", "relay", "retreat", "hold",
]


def build_curriculum() -> list[CurriculumStage]:
    """Progressive difficulty training curriculum."""
    return [
        CurriculumStage(
            name="Phase 1: Basic Survival",
            scenario="configs/scenarios/basic_patrol.yaml",
            episodes=30,
            min_survival_rate=0.7,
            description="Learn to survive against static patrol enemies",
        ),
        CurriculumStage(
            name="Phase 2: Ambush Response",
            scenario="configs/scenarios/ambush_defense.yaml",
            episodes=50,
            min_survival_rate=0.5,
            description="Learn to detect and respond to ambush tactics",
        ),
        CurriculumStage(
            name="Phase 3: EW Adaptation",
            scenario="configs/scenarios/ew_adaptation.yaml",
            episodes=40,
            min_survival_rate=0.4,
            description="Learn to operate under electronic warfare conditions",
        ),
        CurriculumStage(
            name="Phase 4: Stress Test",
            scenario="configs/scenarios/stress_test.yaml",
            episodes=80,
            min_survival_rate=0.3,
            description="Maximum pressure: high lethality, rapid spawning, mixed threats",
        ),
    ]


def train(
    config: AethelredConfig,
    mode: str,
    episodes: int,
    render: bool,
    resume_path: str | None = None,
    noise: bool = True,
    fixed_seed: int | None = None,
    no_adapt: bool = False,
) -> None:
    """Main training loop.

    If ``fixed_seed`` is given, every episode resets to the SAME scenario — useful
    for verifying the agent can actually learn (removes per-episode seed variance).
    """
    log = logging.getLogger("train")

    log.info("=" * 70)
    log.info("  PROJECT AETHELRED - Training Pipeline")
    log.info(f"  Mode: {mode} | Episodes: {episodes} | Device: {config.device}")
    log.info("=" * 70)

    # Reproducible network init / sampling (episodes reseed the env themselves)
    set_seed(42)

    # Build policy from config (so YAML hyperparameters actually take effect)
    policy = TacticalPolicy.build(
        encoder_config=config.state_encoder,
        transformer_config=config.tactical_ai,
        device=config.device,
    )

    # Resume from checkpoint if specified
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=config.device, weights_only=True)
        policy.load_policy_weights(checkpoint["policy_weights"])
        log.info(f"Resumed from {resume_path} (episode {checkpoint.get('episode', '?')})")

    # Build trainer
    trainer = PPOTrainer(
        policy=policy,
        config=config.training,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=f"runs/aethelred_{mode}",
    )

    # Build adaptation engine
    adaptation_engine = AdaptationEngine(config.adaptation, device=config.device)
    opponent_model = OpponentBehaviorModel(device=config.device)
    learning_loop = LearningLoop(
        tactical_policy=policy,
        adaptation_engine=adaptation_engine,
        config=config.learning_loop,
        opponent_model=opponent_model,
    )

    # Safety system
    safety_config = SafetyConfig()
    safety = SafetyManager(
        config=safety_config,
        battlefield_width=config.simulation.battlefield.width,
        battlefield_height=config.simulation.battlefield.height,
    )
    safety_gateway = SafetyExecutionGateway(safety)

    # Set up curriculum if applicable
    if mode == "curriculum":
        trainer.set_curriculum(build_curriculum())
    elif mode == "stress":
        try:
            scenario = load_scenario("configs/scenarios/stress_test.yaml")
            config.simulation = apply_scenario_to_config(config.simulation, scenario)
        except FileNotFoundError:
            log.warning("stress_test.yaml not found, using default config with higher difficulty")
            config.simulation.physics.hit_probability_base = 0.55
            config.simulation.physics.damage_per_hit = 0.5
            config.simulation.threats.initial_threats = 8
            config.simulation.threats.max_threats = 20
            config.simulation.threats.spawn_interval = 20

    # Track best observed reward separately from checkpoint selection.  The
    # selectable checkpoint uses the rolling aggregate below, not one episode.
    best_reward = float("-inf")
    best_survival = 0.0

    render_mode = "human" if render else None

    # Pristine baseline so curriculum stages don't leak settings into each other
    # (each stage's scenario is applied to a fresh copy of the original config).
    pristine_sim = copy.deepcopy(config.simulation)

    for episode in range(episodes):
        # Check curriculum advancement
        if mode == "curriculum":
            trainer.check_curriculum_advance()
            scenario_path = trainer.get_current_scenario()
            if scenario_path:
                try:
                    scenario = load_scenario(scenario_path)
                    config.simulation = apply_scenario_to_config(
                        copy.deepcopy(pristine_sim), scenario
                    )
                except FileNotFoundError:
                    pass

        # Create environment for this episode
        env = AethelredEnv(config=config.simulation, render_mode=render_mode)

        # Reset (fixed scenario if requested, else a fresh seed per episode)
        _obs, info = env.reset(seed=fixed_seed if fixed_seed is not None else 42 + episode)
        policy.reset_history()

        episode_reward = 0.0
        episode_losses = 0
        episode_adaptations_start = learning_loop._total_adaptations
        prev_total_losses = 0
        prev_engagements = 0
        for step in range(config.simulation.max_steps):
            clean_state = env.get_current_state()
            if clean_state is None:
                break

            # Safety: start watchdog + update safety level
            safety.pre_decision(clean_state)

            # The AI acts on a (optionally noisy) observation, never ground truth
            obs_state = safety.inject_training_noise(clean_state) if noise else clean_state
            obs_tensors = BattlefieldStateEncoder.obs_to_tensors(
                env._state_to_obs(obs_state), trainer.device
            )

            # Trainer selects the action AND records the step for PPO
            gym_action = trainer.select_action(obs_tensors)

            # Decode once, then execute only the decision authorised by safety.
            # This prevents watchdog, RTL, emergency-stop, and action-validator
            # corrections from being bypassed by a second Gym-action decode.
            proposal = env._decode_action(gym_action)
            authorised = safety_gateway.authorise(proposal, clean_state)

            # Phase 5 consumption: pre-position recon toward predicted threats
            if not no_adapt and learning_loop.config.enable_prediction:
                env.set_anticipated_threats([
                    p.predicted_position for p in learning_loop.get_predictions(clean_state)
                ])

            obs, reward, terminated, truncated, info = safety_gateway.execute(env, authorised)
            done = terminated or truncated
            episode_reward += reward

            # Preserve Gymnasium's termination/truncation distinction. A time
            # limit receives a value bootstrap from its final observation.
            bootstrap_value = None
            if truncated and not terminated:
                next_obs_tensors = BattlefieldStateEncoder.obs_to_tensors(obs, trainer.device)
                bootstrap_value = trainer.estimate_value(next_obs_tensors)
            trainer.finish_step(
                reward,
                terminated=terminated,
                truncated=truncated,
                bootstrap_value=bootstrap_value,
            )

            # Track losses (for logging)
            new_total = info["total_losses"]
            if new_total > prev_total_losses:
                episode_losses += new_total - prev_total_losses

            # Online adaptation bookkeeping (skipped in pure-PPO --no-adapt mode,
            # which isolates the PPO learner from the adaptation engine).
            if not no_adapt:
                new_state = env.get_current_state()
                if new_state is not None:
                    learning_loop.on_step(new_state, reward)
                if new_total > prev_total_losses:
                    for loss in env.get_losses_since(step - 1):
                        learning_loop.on_loss(loss)
                all_eng = env.get_all_engagements()
                learning_loop.on_engagements(all_eng[prev_engagements:])
                prev_engagements = len(all_eng)
                learning_loop.maybe_adapt()

            prev_total_losses = new_total

            # Run a PPO update once enough experience is collected
            if trainer.ready_to_update:
                trainer.train_on_rollout()

            if done:
                break

        # Flush remaining experience at episode end
        if len(trainer.rollout_buffer) >= 16:
            trainer.train_on_rollout()
        else:
            trainer.rollout_buffer.clear()

        # Episode complete
        episode_adaptations = learning_loop._total_adaptations - episode_adaptations_start
        total_units = config.simulation.swarm.total_units
        alive = info.get("alive_drones", 0)
        survival_rate = alive / max(total_units, 1)

        trainer.metrics.log_episode(
            reward=episode_reward,
            losses=episode_losses,
            survival_rate=survival_rate,
            adaptations=episode_adaptations,
            alive=alive,
            total_units=total_units,
            threats_active=info.get("active_threats", 0),
            steps=step + 1,
        )

        best_reward = max(best_reward, episode_reward)
        rolling_reward = trainer.metrics.avg_reward

        # Checkpoint selection is based on the running evaluation aggregate,
        # avoiding promotion of a model solely because of one lucky rollout.
        if trainer.checkpoint_mgr.is_improved(rolling_reward):
            trainer.checkpoint_mgr.save(
                policy, episode, episode_reward,
                metrics={
                    "survival_rate": survival_rate,
                    "losses": episode_losses,
                    "adaptations": episode_adaptations,
                    "rolling_reward_100": rolling_reward,
                },
                value_head=trainer.value_head,
                optimizer=trainer.optimizer,
                selection_score=rolling_reward,
                selection_metric="rolling_reward_100",
            )

        best_survival = max(best_survival, survival_rate)

        # Periodic checkpoint
        if episode % 25 == 0 and episode > 0:
            trainer.checkpoint_mgr.save(
                policy, episode, episode_reward,
                metrics={"survival_rate": survival_rate},
            )

        # Log
        if episode % 5 == 0:
            stage_name = ""
            if mode == "curriculum" and trainer._curriculum:
                idx = min(trainer._current_stage, len(trainer._curriculum) - 1)
                stage_name = f" [{trainer._curriculum[idx].name}]"

            log.info(
                f"Ep {episode:4d}{stage_name} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Alive: {alive:2d}/{total_units} ({survival_rate:.0%}) | "
                f"Losses: {episode_losses} | "
                f"Adapt: {episode_adaptations} | "
                f"Best: {best_reward:.1f} | "
                f"Safety: {safety.level.value}"
            )

        env.close()

    # Final report
    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Episodes: {episodes}")
    log.info(f"  Best reward: {best_reward:.2f}")
    log.info(f"  Best survival: {best_survival:.0%}")
    log.info(f"  Avg reward (last 100): {trainer.metrics.avg_reward:.2f}")
    log.info(f"  Avg survival (last 100): {trainer.metrics.avg_survival:.0%}")
    log.info(f"  Known threats: {len(adaptation_engine.known_threat_types)}")
    log.info(f"  Total adaptations: {learning_loop._total_adaptations}")
    log.info(f"  Safety events: {len(safety.events)}")
    log.info(f"  Watchdog timeouts: {safety.watchdog.total_timeouts}")
    log.info("")
    log.info(f"  Checkpoints saved to: {config.checkpoint_dir}/")
    log.info(f"  TensorBoard logs: runs/aethelred_{mode}/")
    log.info("=" * 70)

    trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aethelred Tactical AI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument(
        "--mode", type=str, default="online",
        choices=["online", "curriculum", "stress"],
        help="Training mode: online (single scenario), curriculum (progressive), stress (max difficulty)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-noise", action="store_true", help="Disable sensor noise injection")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed scenario seed every episode (for learning sanity checks)")
    parser.add_argument("--no-adapt", action="store_true",
                        help="Disable the online adaptation engine (train pure PPO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = AethelredConfig()
    if args.config:
        config = AethelredConfig.from_yaml(args.config)
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config.simulation = apply_scenario_to_config(config.simulation, scenario)
    if args.device:
        config.device = args.device

    train(
        config=config,
        mode=args.mode,
        episodes=args.episodes,
        render=args.render,
        resume_path=args.resume,
        noise=not args.no_noise,
        fixed_seed=args.seed,
        no_adapt=args.no_adapt,
    )


if __name__ == "__main__":
    main()
