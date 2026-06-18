"""
Stress-test the Aethelred system under extreme conditions.

Tests:
  1. Ambush scenario (high lethality, concealed positions)
  2. Stress scenario (maximum threats, rapid spawn, aggressive enemies)
  3. Adaptation resilience (new threat types mid-episode)
  4. Safety system validation (geofence, watchdog, RTL)

Usage:
    python scripts/stress_test.py
    python scripts/stress_test.py --scenario ambush --episodes 10
    python scripts/stress_test.py --scenario stress --episodes 5
    python scripts/stress_test.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.adaptation.opponent_model import OpponentBehaviorModel
from aethelred.config.settings import AethelredConfig
from aethelred.deployment.safety import SafetyConfig, SafetyManager
from aethelred.learning.learning_loop import LearningLoop
from aethelred.simulation.environment import AethelredEnv
from aethelred.simulation.scenarios import apply_scenario_to_config, load_scenario
from aethelred.swarm.coordinator import SwarmCoordinator
from aethelred.swarm.mother_drone import MotherDrone
from aethelred.swarm.swarm_unit import SwarmUnit
from aethelred.tactical_ai.policy import TacticalPolicy
from aethelred.utils.seeding import set_seed


def run_scenario_test(
    scenario_name: str,
    scenario_path: str,
    episodes: int,
    logger: logging.Logger,
) -> dict:
    """Run a scenario and collect metrics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  STRESS TEST: {scenario_name}")
    logger.info(f"{'='*60}")

    set_seed(100)

    config = AethelredConfig()
    try:
        scenario = load_scenario(scenario_path)
        config.simulation = apply_scenario_to_config(config.simulation, scenario)
    except FileNotFoundError:
        logger.warning(f"Scenario {scenario_path} not found, using defaults with high difficulty")
        config.simulation.physics.hit_probability_base = 0.5
        config.simulation.physics.damage_per_hit = 0.45
        config.simulation.threats.initial_threats = 6
        config.simulation.threats.max_threats = 15

    # Build components
    policy = TacticalPolicy.build(
        encoder_config=config.state_encoder,
        transformer_config=config.tactical_ai,
        device=config.device,
    )
    policy.set_world_size(
        config.simulation.battlefield.width, config.simulation.battlefield.height
    )
    adaptation_engine = AdaptationEngine(config.adaptation, device=config.device)
    opponent_model = OpponentBehaviorModel(device=config.device)
    learning_loop = LearningLoop(
        tactical_policy=policy,
        adaptation_engine=adaptation_engine,
        config=config.learning_loop,
        opponent_model=opponent_model,
    )

    safety = SafetyManager(
        config=SafetyConfig(),
        battlefield_width=config.simulation.battlefield.width,
        battlefield_height=config.simulation.battlefield.height,
    )

    results = {
        "scenario": scenario_name,
        "episodes": episodes,
        "total_losses": 0,
        "total_adaptations": 0,
        "total_safety_events": 0,
        "episode_results": [],
        "threat_types_seen": set(),
        "min_survival": 1.0,
        "max_losses_single_episode": 0,
        "avg_decision_time_ms": 0.0,
    }

    decision_times = []

    for episode in range(episodes):
        env = AethelredEnv(config=config.simulation)
        coordinator = SwarmCoordinator()
        mother = MotherDrone(
            tactical_policy=policy,
            learning_loop=learning_loop,
            swarm_coordinator=coordinator,
        )

        obs, info = env.reset(seed=100 + episode)
        policy.reset_history()

        state = env.get_current_state()
        if state:
            for drone in state.active_friendlies:
                coordinator.register_unit(SwarmUnit(drone))

        ep_reward = 0.0
        ep_losses = 0
        ep_adapt_start = learning_loop._total_adaptations
        prev_total = 0
        prev_engagements = 0

        for step in range(config.simulation.max_steps):
            state = env.get_current_state()
            if state is None:
                break

            state = safety.pre_decision(state)

            # Time the decision
            t0 = time.perf_counter()
            decision = mother.tick(state)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            decision_times.append(dt_ms)

            decision = safety.post_decision(decision, state)

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
                gym_action["action_type"] = (
                    action_types.index(act.action_type.value)
                    if act.action_type.value in action_types else 0
                )
                if act.target_position:
                    gym_action["target_position"] = [
                        act.target_position.x / config.simulation.battlefield.width,
                        act.target_position.y / config.simulation.battlefield.height,
                    ]

            # Phase 5 consumption: pre-position recon toward predicted threats
            if config.learning_loop.enable_prediction:
                env.set_anticipated_threats([
                    p.predicted_position for p in learning_loop.get_predictions(state)
                ])

            obs, reward, terminated, truncated, info = env.step(gym_action)
            ep_reward += reward

            new_total = info["total_losses"]
            if new_total > prev_total:
                for loss in env.get_losses_since(step - 1):
                    mother.handle_unit_loss(loss)
                    ep_losses += 1
                    results["threat_types_seen"].add(loss.cause_of_loss.value)
            prev_total = new_total

            # Record threats neutralized this step (updates threat mastery)
            all_eng = env.get_all_engagements()
            learning_loop.on_engagements(all_eng[prev_engagements:])
            prev_engagements = len(all_eng)

            if terminated or truncated:
                break

        ep_adaptations = learning_loop._total_adaptations - ep_adapt_start
        total_units = config.simulation.swarm.total_units
        alive = info.get("alive_drones", 0)
        survival = alive / max(total_units, 1)

        results["total_losses"] += ep_losses
        results["total_adaptations"] += ep_adaptations
        results["total_safety_events"] += len(safety.events)
        results["min_survival"] = min(results["min_survival"], survival)
        results["max_losses_single_episode"] = max(results["max_losses_single_episode"], ep_losses)

        ep_result = {
            "episode": episode,
            "reward": ep_reward,
            "losses": ep_losses,
            "adaptations": ep_adaptations,
            "alive": alive,
            "survival": survival,
            "steps": step + 1,
            "safety_level": safety.level.value,
        }
        results["episode_results"].append(ep_result)

        logger.info(
            f"  Ep {episode+1:2d}/{episodes}: "
            f"Alive={alive:2d}/{total_units} ({survival:.0%}) | "
            f"Losses={ep_losses:2d} | "
            f"Adaptations={ep_adaptations} | "
            f"Reward={ep_reward:.1f} | "
            f"Safety={safety.level.value}"
        )

        env.close()

    # Compute summary
    import numpy as np
    if decision_times:
        dt_arr = np.array(decision_times)
        results["avg_decision_time_ms"] = float(dt_arr.mean())
        results["p95_decision_time_ms"] = float(np.percentile(dt_arr, 95))
        results["p99_decision_time_ms"] = float(np.percentile(dt_arr, 99))

    avg_survival = sum(r["survival"] for r in results["episode_results"]) / max(episodes, 1)
    avg_losses = results["total_losses"] / max(episodes, 1)

    logger.info(f"\n  --- {scenario_name} RESULTS ---")
    logger.info(f"  Avg survival rate: {avg_survival:.0%}")
    logger.info(f"  Avg losses/episode: {avg_losses:.1f}")
    logger.info(f"  Max losses (single ep): {results['max_losses_single_episode']}")
    logger.info(f"  Min survival: {results['min_survival']:.0%}")
    logger.info(f"  Total adaptations: {results['total_adaptations']}")
    logger.info(f"  Threat types seen: {results['threat_types_seen']}")
    logger.info(f"  Avg decision time: {results['avg_decision_time_ms']:.2f}ms")
    logger.info(f"  p95 decision time: {results.get('p95_decision_time_ms', 0):.2f}ms")
    logger.info(f"  Safety events: {results['total_safety_events']}")
    logger.info(f"  Known threats: {len(adaptation_engine.known_threat_types)}")

    threat_summary = adaptation_engine.get_threat_summary()
    if threat_summary:
        logger.info("  Threat knowledge:")
        for tt, data in threat_summary.items():
            logger.info(
                f"    {tt}: encounters={data['encounters']}, "
                f"lethality={data['lethality']:.2f}, "
                f"mastered={'YES' if data['mastered'] else 'no'}"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress Test Aethelred")
    parser.add_argument("--scenario", type=str, default="ambush",
                        choices=["ambush", "stress", "ew"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("stress_test")

    logger.info("=" * 70)
    logger.info("  PROJECT AETHELRED - Stress Test Suite")
    logger.info("=" * 70)

    scenarios = {
        "ambush": ("Ambush Defense", "configs/scenarios/ambush_defense.yaml"),
        "stress": ("Maximum Stress", "configs/scenarios/stress_test.yaml"),
        "ew": ("EW Adaptation", "configs/scenarios/ew_adaptation.yaml"),
    }

    all_results = {}

    if args.all:
        for key, (name, path) in scenarios.items():
            all_results[key] = run_scenario_test(name, path, args.episodes, logger)
    else:
        name, path = scenarios[args.scenario]
        all_results[args.scenario] = run_scenario_test(name, path, args.episodes, logger)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("  STRESS TEST SUMMARY")
    logger.info("=" * 70)
    for key, results in all_results.items():
        avg_surv = sum(r["survival"] for r in results["episode_results"]) / max(len(results["episode_results"]), 1)
        logger.info(
            f"  {results['scenario']:20s} | "
            f"Survival={avg_surv:.0%} | "
            f"Losses={results['total_losses']:3d} | "
            f"Adaptations={results['total_adaptations']:2d} | "
            f"Decision={results['avg_decision_time_ms']:.1f}ms"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
