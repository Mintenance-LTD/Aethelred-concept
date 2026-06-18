"""Environment smoke + reproducibility tests."""

from __future__ import annotations

import numpy as np

from aethelred.config.settings import SimulationConfig
from aethelred.simulation.environment import AethelredEnv


def _small_config() -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.max_steps = 30
    return cfg


def _fixed_action() -> dict:
    return {
        "action_type": 1,
        "target_position": np.array([0.5, 0.5], dtype=np.float32),
        "priority": np.array([0.5], dtype=np.float32),
        "formation": 0,
        "target_index": 0,
    }


def _run(seed: int) -> list[float]:
    env = AethelredEnv(config=_small_config())
    env.reset(seed=seed)
    rewards = []
    for _ in range(30):
        _, reward, terminated, truncated, _ = env.step(_fixed_action())
        rewards.append(round(reward, 6))
        if terminated or truncated:
            break
    env.close()
    return rewards


def test_env_runs():
    rewards = _run(123)
    assert len(rewards) > 0


def test_runs_are_reproducible_with_same_seed():
    # Identical seed + identical actions must yield identical trajectories.
    # (Regression test for the unseeded global-`random` combat resolution.)
    assert _run(7) == _run(7)


def test_different_seeds_differ():
    # Seeding must actually influence the sim: different seeds -> different
    # initial threat spawn positions.
    def _spawn_positions(seed: int) -> list[tuple[float, float]]:
        env = AethelredEnv(config=_small_config())
        env.reset(seed=seed)
        state = env.get_current_state()
        env.close()
        return [(round(t.position.x, 3), round(t.position.y, 3)) for t in state.threats]

    assert _spawn_positions(1) != _spawn_positions(2)
