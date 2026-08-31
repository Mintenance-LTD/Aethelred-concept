"""Tests for safety systems."""

from __future__ import annotations

import numpy as np
import pytest

from aethelred.config.settings import AethelredConfig, ConfigurationError
from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.enums import DroneRole, TacticalActionType
from aethelred.core.models import BattlefieldState, DroneState, Vec2
from aethelred.deployment.safety import (
    GeofenceGuard,
    SafetyConfig,
    SafetyExecutionGateway,
    SafetyManager,
    SensorNoiseInjector,
)


def test_noise_injection_does_not_mutate_ground_truth():
    drone = DroneState(role=DroneRole.ENGAGE, position=Vec2(x=100.0, y=100.0))
    state = BattlefieldState(friendly_units=[drone])
    injector = SensorNoiseInjector(SafetyConfig(), rng=np.random.default_rng(0))

    noisy = injector.inject_state_noise(state)

    assert noisy is not state
    assert state.friendly_units[0].position.x == 100.0  # original untouched
    assert state.friendly_units[0].position.y == 100.0


def test_geofence_clamps_out_of_bounds():
    guard = GeofenceGuard(width=1000.0, height=1000.0, margin=50.0, hard_limit=20.0)
    pos, violated = guard.validate_position(Vec2(x=-5.0, y=500.0))
    assert violated
    assert pos.x >= 20.0

    pos2, violated2 = guard.validate_position(Vec2(x=500.0, y=500.0))
    assert not violated2
    assert pos2.x == 500.0


class _RecordingExecutor:
    def __init__(self) -> None:
        self.executed: TacticalDecision | None = None

    def step_decision(
        self, decision: TacticalDecision
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        self.executed = decision
        return {}, 0.0, False, False, {}


def test_safety_gateway_executes_emergency_hold_not_raw_proposal():
    drone = DroneState(role=DroneRole.ENGAGE, position=Vec2(x=100.0, y=100.0))
    state = BattlefieldState(friendly_units=[drone])
    proposal = TacticalDecision(
        timestep=state.timestep,
        actions=[
            TacticalAction(
                action_type=TacticalActionType.ENGAGE,
                target_unit_id=drone.id,
                target_position=Vec2(x=900.0, y=900.0),
            )
        ],
    )
    safety = SafetyManager(SafetyConfig())
    safety.emergency_stop()
    gateway = SafetyExecutionGateway(safety)
    executor = _RecordingExecutor()

    authorised = gateway.authorise(proposal, state)
    gateway.execute(executor, authorised)

    assert executor.executed is authorised.decision
    assert executor.executed is not proposal
    assert [action.action_type for action in executor.executed.actions] == [TacticalActionType.HOLD]


def test_config_rejects_unknown_keys_and_conflicting_devices():
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        AethelredConfig._from_dict({"simulaton": {}})

    with pytest.raises(ConfigurationError, match="must match"):
        AethelredConfig._from_dict({"device": "cpu", "training": {"device": "cuda"}})


def test_root_device_is_propagated_to_trainer_device():
    config = AethelredConfig._from_dict({"device": "cuda"})
    assert config.device == "cuda"
    assert config.training.device == "cuda"


def test_yaml_config_requires_supported_schema_version(tmp_path):
    missing_version = tmp_path / "missing-version.yaml"
    missing_version.write_text("device: cpu\n", encoding="utf-8")
    unsupported_version = tmp_path / "unsupported-version.yaml"
    unsupported_version.write_text("schema_version: 2\ndevice: cpu\n", encoding="utf-8")
    supported_version = tmp_path / "supported-version.yaml"
    supported_version.write_text("schema_version: 1\ndevice: cpu\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="declare schema_version"):
        AethelredConfig.from_yaml(missing_version)
    with pytest.raises(ConfigurationError, match="Unsupported configuration schema_version"):
        AethelredConfig.from_yaml(unsupported_version)

    assert AethelredConfig.from_yaml(supported_version).schema_version == 1


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"simulation": {"physics": {"hit_probability_base": 1.1}}}, "between 0 and 1"),
        ({"simulation": {"threats": {"initial_threats": 11, "max_threats": 10}}}, "initial_threats"),
        ({"tactical_ai": {"hidden_dim": 250, "num_heads": 8}}, "divisible"),
        ({"tactical_ai": {"state_embed_dim": 128}}, "state_embed_dim must match"),
        ({"training": {"gamma": -0.1}}, "between 0 and 1"),
    ],
)
def test_config_rejects_unsafe_numeric_and_structural_values(data, message):
    with pytest.raises(ConfigurationError, match=message):
        AethelredConfig._from_dict(data)
