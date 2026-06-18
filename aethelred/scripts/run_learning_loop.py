"""
Run the full 5-phase learning loop in simulation.
This is the primary demonstration of Aethelred's core innovation.

Usage:
    python scripts/run_learning_loop.py
    python scripts/run_learning_loop.py --episodes 10 --render
    python scripts/run_learning_loop.py --config configs/scenarios/ambush_defense.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.adaptation.opponent_model import OpponentBehaviorModel
from aethelred.config.settings import AethelredConfig
from aethelred.learning.learning_loop import LearningLoop
from aethelred.simulation.environment import AethelredEnv
from aethelred.simulation.scenarios import apply_scenario_to_config, load_scenario
from aethelred.swarm.coordinator import SwarmCoordinator
from aethelred.swarm.mother_drone import MotherDrone
from aethelred.swarm.swarm_unit import SwarmUnit
from aethelred.tactical_ai.policy import TacticalPolicy
from aethelred.utils.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aethelred Learning Loop")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("aethelred")

    set_seed(args.seed)

    # Load config
    config = AethelredConfig()
    if args.config:
        config = AethelredConfig.from_yaml(args.config)
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config.simulation = apply_scenario_to_config(config.simulation, scenario)

    logger.info("=" * 70)
    logger.info("  PROJECT AETHELRED - The Divine General")
    logger.info("  Adaptive Autonomous Swarm Intelligence System")
    logger.info("  Learning Loop Demonstration")
    logger.info("=" * 70)

    # Build all components
    logger.info("Building tactical AI...")
    policy = TacticalPolicy.build(
        encoder_config=config.state_encoder,
        transformer_config=config.tactical_ai,
        device=config.device,
    )
    policy.set_world_size(
        config.simulation.battlefield.width, config.simulation.battlefield.height
    )

    logger.info("Building adaptation engine...")
    adaptation_engine = AdaptationEngine(config.adaptation, device=config.device)
    opponent_model = OpponentBehaviorModel(device=config.device)

    logger.info("Building learning loop...")
    learning_loop = LearningLoop(
        tactical_policy=policy,
        adaptation_engine=adaptation_engine,
        config=config.learning_loop,
        opponent_model=opponent_model,
    )

    # Create environment
    render_mode = "human" if args.render else None
    env = AethelredEnv(config=config.simulation, render_mode=render_mode)

    logger.info(f"Swarm size: {config.simulation.swarm.total_units} units")
    logger.info(f"Episodes: {args.episodes}")
    logger.info("")

    # Track metrics across episodes
    all_rewards = []
    all_losses = []
    all_adaptations = []

    for episode in range(args.episodes):
        logger.info(f"{'='*50}")
        logger.info(f"EPISODE {episode + 1}/{args.episodes}")
        logger.info(f"{'='*50}")

        # Build swarm coordinator for this episode
        coordinator = SwarmCoordinator()
        mother = MotherDrone(
            tactical_policy=policy,
            learning_loop=learning_loop,
            swarm_coordinator=coordinator,
        )

        # Reset environment and policy history
        obs, info = env.reset(seed=args.seed + episode)
        policy.reset_history()

        # Register swarm units from environment
        state = env.get_current_state()
        if state:
            for drone in state.active_friendlies:
                unit = SwarmUnit(drone)
                coordinator.register_unit(unit)

        episode_reward = 0.0
        episode_losses = 0
        episode_adaptations = 0
        prev_total_losses = 0
        prev_engagements = 0

        for step in range(config.simulation.max_steps):
            # Get current state and run mother drone tick
            state = env.get_current_state()
            if state is None:
                break

            decision = mother.tick(state)

            # Convert decision to gym action
            gym_action = {
                "action_type": 0,
                "target_position": [0.5, 0.5],
                "priority": [0.5],
                "formation": 0,
                "target_index": 0,
            }
            if decision.actions:
                act = decision.actions[0]
                action_types = ["move", "engage", "evade", "formation_change",
                                "recon", "relay", "retreat", "hold"]
                gym_action["action_type"] = action_types.index(act.action_type.value) \
                    if act.action_type.value in action_types else 0
                if act.target_position:
                    gym_action["target_position"] = [
                        act.target_position.x / 1000.0,
                        act.target_position.y / 1000.0,
                    ]

            # Phase 5 consumption: pre-position recon toward predicted threats
            if config.learning_loop.enable_prediction:
                env.set_anticipated_threats([
                    p.predicted_position for p in learning_loop.get_predictions(state)
                ])

            obs, reward, terminated, truncated, info = env.step(gym_action)
            episode_reward += reward

            # Check for new losses and forward to mother
            new_total = info["total_losses"]
            if new_total > prev_total_losses:
                for loss in env.get_losses_since(step - 1):
                    mother.handle_unit_loss(loss)
                    episode_losses += 1
            prev_total_losses = new_total

            # Record threats neutralized this step (updates threat mastery)
            all_eng = env.get_all_engagements()
            learning_loop.on_engagements(all_eng[prev_engagements:])
            prev_engagements = len(all_eng)

            # Count adaptations
            current_adaptations = learning_loop._total_adaptations
            if current_adaptations > sum(all_adaptations) + episode_adaptations:
                episode_adaptations = current_adaptations - sum(all_adaptations)

            # Periodic logging
            if step % 100 == 0 and step > 0:
                logger.info(
                    f"  Step {step:4d} | Alive: {info['alive_drones']:2d} | "
                    f"Threats: {info['active_threats']:2d} | "
                    f"Losses: {episode_losses} | "
                    f"Adaptations: {episode_adaptations} | "
                    f"Reward: {episode_reward:.1f}"
                )

            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        all_losses.append(episode_losses)
        all_adaptations.append(episode_adaptations)

        logger.info(f"\n  Episode {episode + 1} Summary:")
        logger.info(f"    Reward: {episode_reward:.2f}")
        logger.info(f"    Losses: {episode_losses}")
        logger.info(f"    Adaptations: {episode_adaptations}")
        logger.info(f"    Surviving: {info['alive_drones']}/{config.simulation.swarm.total_units}")
        logger.info(f"    Known threats: {len(adaptation_engine.known_threat_types)}")

        threat_summary = adaptation_engine.get_threat_summary()
        if threat_summary:
            logger.info("    Threat knowledge:")
            for tt, data in threat_summary.items():
                logger.info(
                    f"      {tt}: encounters={data['encounters']}, "
                    f"lethality={data['lethality']:.2f}, "
                    f"counter_eff={data['counter_eff']:.2f}, "
                    f"mastered={'YES' if data['mastered'] else 'no'}"
                )

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  LEARNING LOOP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total episodes: {args.episodes}")
    logger.info(f"  Avg reward: {sum(all_rewards)/len(all_rewards):.2f}")
    logger.info(f"  Total losses: {sum(all_losses)}")
    logger.info(f"  Total adaptations: {sum(all_adaptations)}")
    logger.info(f"  Known threat types: {len(adaptation_engine.known_threat_types)}")
    logger.info("")
    logger.info('  "The wheel turns. The Divine General learns. The swarm evolves."')
    logger.info("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
