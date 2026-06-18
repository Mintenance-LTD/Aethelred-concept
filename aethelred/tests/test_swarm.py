"""Tests for swarm behaviour and threat-mastery wiring (iteration 2 fixes)."""

from __future__ import annotations

from uuid import uuid4

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.config.settings import AdaptationConfig, SimulationConfig
from aethelred.core.enums import DroneRole, EngagementOutcome, TacticalActionType, ThreatType
from aethelred.core.events import EngagementEvent
from aethelred.core.models import DroneState, Vec2
from aethelred.simulation.environment import AethelredEnv


def test_decode_action_is_role_heterogeneous():
    """One ENGAGE command must expand into different per-role actions, not N copies."""
    cfg = SimulationConfig()
    cfg.max_steps = 10
    env = AethelredEnv(config=cfg)
    env.reset(seed=3)

    action = {
        "action_type": list(TacticalActionType).index(TacticalActionType.ENGAGE),
        "target_position": [0.5, 0.5],
        "priority": [0.9],
        "formation": 0,
        "target_index": 0,
    }
    decision = env._decode_action(action)

    types_used = {a.action_type for a in decision.actions}
    assert len(types_used) > 1, "swarm is still homogeneous (one action for all units)"

    # Recon units should be reconning, not engaging
    recon_actions = {
        a.action_type for a in decision.actions
        if env.entity_manager.drones[a.target_unit_id].role == DroneRole.RECON
    }
    assert recon_actions <= {TacticalActionType.RECON, TacticalActionType.EVADE}

    # Engage units distribute fire across multiple threats
    engage_targets = {a.target_threat_id for a in decision.actions if a.target_threat_id is not None}
    assert len(engage_targets) >= 2, "engage units not distributed across threats"
    env.close()


def test_recon_prepositions_toward_anticipated_threats():
    """Opponent-model predictions must actually steer recon (not be ignored)."""
    cfg = SimulationConfig()
    cfg.max_steps = 10
    env = AethelredEnv(config=cfg)
    env.reset(seed=5)

    anticipated = Vec2(x=320.0, y=520.0)  # near the rally point -> recon stays linked
    env.set_anticipated_threats([anticipated])

    action = {
        "action_type": list(TacticalActionType).index(TacticalActionType.MOVE),
        "target_position": [0.9, 0.9],
        "priority": [0.5],
        "formation": 0,
        "target_index": 0,
    }
    decision = env._decode_action(action)
    recon = [
        a for a in decision.actions
        if env.entity_manager.drones[a.target_unit_id].role == DroneRole.RECON
        and a.action_type == TacticalActionType.RECON
    ]
    assert recon, "no connected recon units to pre-position"
    assert any(
        a.target_position is not None
        and abs(a.target_position.x - anticipated.x) < 1e-6
        and abs(a.target_position.y - anticipated.y) < 1e-6
        for a in recon
    )
    env.close()


def test_distant_drone_loses_comms_and_acts_autonomously():
    """A drone far from the command node drops to autonomy (uses SwarmUnit)."""
    cfg = SimulationConfig()
    env = AethelredEnv(config=cfg)
    env.reset(seed=1)

    far = DroneState(
        role=DroneRole.ENGAGE,
        position=Vec2(x=cfg.battlefield.width * 0.95, y=cfg.battlefield.height * 0.5),
    )
    assert env._comm_status_for(far) == "lost"
    action = env._autonomous_action(far, "lost", [])
    assert action.target_unit_id == far.id
    env.close()


def test_record_neutralizations_updates_mastery():
    eng = AdaptationEngine(AdaptationConfig())
    events = [
        EngagementEvent(
            timestep=1, friendly_id=uuid4(), threat_id=uuid4(),
            threat_type=ThreatType.SMALL_ARMS,
            outcome=EngagementOutcome.TARGET_DESTROYED,
            friendly_health_before=1.0, friendly_health_after=1.0,
            threat_health_before=1.0, threat_health_after=0.0, distance=10.0,
        )
        for _ in range(2)
    ]
    eng.record_neutralizations(events)
    profile = eng.threat_classifier.profiles[ThreatType.SMALL_ARMS]
    assert profile.neutralized_count == 2
    assert profile.counter_effectiveness > 0.0
