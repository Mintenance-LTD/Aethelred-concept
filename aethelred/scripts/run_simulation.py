"""
Run an Aethelred tactical simulation.

Usage:
    python scripts/run_simulation.py --render
    python scripts/run_simulation.py --config configs/scenarios/ambush_defense.yaml --render
    python scripts/run_simulation.py --steps 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aethelred.config.settings import AethelredConfig
from aethelred.simulation.environment import AethelredEnv
from aethelred.simulation.scenarios import apply_scenario_to_config, load_scenario
from aethelred.utils.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aethelred simulation")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario YAML path")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument("--steps", type=int, default=None, help="Max simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    # Load config
    if args.config:
        config = AethelredConfig.from_yaml(args.config)
    else:
        config = AethelredConfig()

    # Apply scenario overrides
    if args.scenario:
        scenario = load_scenario(args.scenario)
        config.simulation = apply_scenario_to_config(config.simulation, scenario)

    if args.steps:
        config.simulation.max_steps = args.steps

    # Create environment
    render_mode = "human" if args.render else None
    env = AethelredEnv(config=config.simulation, render_mode=render_mode)

    logger.info("=" * 60)
    logger.info("PROJECT AETHELRED - Tactical Simulation")
    logger.info("=" * 60)
    logger.info(f"Battlefield: {config.simulation.battlefield.width}x{config.simulation.battlefield.height}")
    logger.info(f"Swarm: {config.simulation.swarm.total_units} units")
    logger.info(f"Max steps: {config.simulation.max_steps}")
    logger.info(f"Render: {args.render}")
    logger.info("")

    # Run simulation with random policy
    obs, info = env.reset(seed=args.seed)

    total_reward = 0.0
    for step in range(config.simulation.max_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            logger.info(
                f"Step {step:4d} | "
                f"Alive: {info['alive_drones']:2d} | "
                f"Threats: {info['active_threats']:2d} | "
                f"Losses: {info['total_losses']:2d} | "
                f"Objectives: {info['objectives_completed']} | "
                f"Reward: {total_reward:.2f}"
            )

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            logger.info(f"\nSimulation {reason} at step {step}")
            break

    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Total steps: {info['step']}")
    logger.info(f"Total reward: {total_reward:.2f}")
    logger.info(f"Surviving drones: {info['alive_drones']}")
    logger.info(f"Total losses: {info['total_losses']}")
    logger.info(f"Objectives completed: {info['objectives_completed']}")
    logger.info("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
