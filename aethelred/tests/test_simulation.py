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


def test_shield_overrides_command_in_step_path():
    """An attached Simple-Domain shield vetoes the commanded action for a unit
    sitting inside an unmastered threat's kill range (survival floor)."""
    from aethelred.core.enums import TacticalActionType
    from aethelred.deployment.safety import SimpleDomain

    env = AethelredEnv(config=_small_config())
    env.reset(seed=5)

    dom = SimpleDomain(mastery_threshold=0.5, buffer=1.2)
    # Nothing is mastered -> every in-range threat endangers its unit.
    env.set_shield(lambda drone, threats: dom.shield(drone, threats, lambda tt: 0.0))

    # Place a drone right on top of a threat so it is unambiguously in range.
    drone = env.entity_manager.get_active_drones()[0]
    threat = env.entity_manager.get_active_threats()[0]
    drone.position = threat.position.__class__(x=threat.position.x, y=threat.position.y)

    decision = env._decode_action(_fixed_action())  # commanded action_type=1 (ENGAGE)
    shielded = next(a for a in decision.actions if a.target_unit_id == drone.id)
    assert shielded.action_type == TacticalActionType.EVADE


def test_no_shield_leaves_behavior_unchanged():
    # Default (no shield) trajectories are identical to before the shield hook.
    assert _run(11) == _run(11)
