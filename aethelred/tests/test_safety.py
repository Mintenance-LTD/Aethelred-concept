"""Tests for safety systems."""

from __future__ import annotations

import numpy as np

from aethelred.core.enums import DroneRole
from aethelred.core.models import BattlefieldState, DroneState, Vec2
from aethelred.deployment.safety import GeofenceGuard, SafetyConfig, SensorNoiseInjector


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
