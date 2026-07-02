"""Tests for the divine adaptive engine additions:

  - CounterBank: outcome-driven, per-threat counter memory that needs turns of
    the wheel before it trusts a counter and never forgets one.
  - SimpleDomain: the survival-floor reflex that keeps units out of the kill
    range of unmastered threats.
  - AdaptationEngine wiring of the two.
  - The C5 fix: the learning loop can adapt WITHOUT overwriting policy weights.
"""

from __future__ import annotations

from uuid import uuid4

import torch

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.adaptation.counter_bank import CounterBank
from aethelred.config.settings import AdaptationConfig, LearningLoopConfig
from aethelred.core.enums import (
    DroneRole,
    EngagementOutcome,
    TacticalActionType,
    ThreatCategory,
    ThreatType,
)
from aethelred.core.events import EngagementEvent, LossEvent
from aethelred.core.models import DroneState, ThreatState, Vec2
from aethelred.deployment.safety import SimpleDomain
from aethelred.learning.learning_loop import LearningLoop
from aethelred.tactical_ai.policy import TacticalPolicy

_ACTION_INDEX = {a: i for i, a in enumerate(TacticalActionType)}
_ENGAGE = _ACTION_INDEX[TacticalActionType.ENGAGE]
_RETREAT = _ACTION_INDEX[TacticalActionType.RETREAT]


# --- CounterBank ---------------------------------------------------------

def test_counter_bank_needs_turns_before_it_trusts_a_counter():
    bank = CounterBank(min_turns=3)
    # Two turns is below the threshold -> no trusted recommendation yet.
    bank.record_outcome(ThreatType.SMALL_ARMS, _ENGAGE, 1.0)
    bank.record_outcome(ThreatType.SMALL_ARMS, _ENGAGE, 1.0)
    idx, conf = bank.recommend(ThreatType.SMALL_ARMS)
    assert idx is None and conf == 0.0

    bank.record_outcome(ThreatType.SMALL_ARMS, _ENGAGE, 1.0)
    idx, conf = bank.recommend(ThreatType.SMALL_ARMS)
    assert idx == _ENGAGE and conf > 0.0


def test_counter_bank_learns_the_state_dependent_counter():
    bank = CounterBank(min_turns=3)
    # Winnable small-arms: engaging pays, retreating does not.
    for _ in range(5):
        bank.record_outcome(ThreatType.SMALL_ARMS, _ENGAGE, 1.0)
        bank.record_outcome(ThreatType.SMALL_ARMS, _RETREAT, -0.2)
    # Deadly AA: engaging is fatal, retreating survives.
    for _ in range(5):
        bank.record_outcome(ThreatType.AA_MISSILE, _ENGAGE, -1.0)
        bank.record_outcome(ThreatType.AA_MISSILE, _RETREAT, 0.3)

    assert bank.recommend(ThreatType.SMALL_ARMS)[0] == _ENGAGE
    assert bank.recommend(ThreatType.AA_MISSILE)[0] == _RETREAT
    # The prior nudges toward the learned counter for each context.
    assert bank.action_prior(ThreatType.SMALL_ARMS)[_ENGAGE] > 0
    assert bank.action_prior(ThreatType.AA_MISSILE)[_RETREAT] > 0


def test_counter_bank_never_forgets():
    bank = CounterBank(min_turns=3)
    for _ in range(4):
        bank.record_outcome(ThreatType.SMALL_ARMS, _ENGAGE, 1.0)
    mastered_before = bank.mastery(ThreatType.SMALL_ARMS)
    # Learn an entirely different threat afterwards.
    for _ in range(4):
        bank.record_outcome(ThreatType.AA_MISSILE, _RETREAT, 1.0)
    # The earlier counter is untouched — a table cannot catastrophically forget.
    assert bank.recommend(ThreatType.SMALL_ARMS)[0] == _ENGAGE
    assert bank.mastery(ThreatType.SMALL_ARMS) == mastered_before


# --- SimpleDomain --------------------------------------------------------

def _drone(x: float, y: float) -> DroneState:
    return DroneState(role=DroneRole.ENGAGE, position=Vec2(x=x, y=y))


def _threat(x: float, y: float, rng: float, tt=ThreatType.AA_MISSILE) -> ThreatState:
    return ThreatState(
        threat_type=tt, category=ThreatCategory.KINETIC,
        position=Vec2(x=x, y=y), estimated_range=rng,
    )


def test_simple_domain_shields_unit_inside_unmastered_kill_range():
    dom = SimpleDomain(mastery_threshold=0.5, buffer=1.2)
    drone = _drone(500, 500)
    threat = _threat(520, 500, rng=140.0)  # 20m away, well inside 140m kill range

    override = dom.shield(drone, [threat], mastery_of=lambda tt: 0.0)
    assert override is not None
    assert override.action_type == TacticalActionType.EVADE
    # It pushes the unit away from the threat, past the kill range.
    new_dist = override.target_position.distance_to(threat.position)
    assert new_dist > threat.estimated_range


def test_simple_domain_does_not_shield_a_mastered_threat():
    dom = SimpleDomain(mastery_threshold=0.5)
    drone = _drone(500, 500)
    threat = _threat(520, 500, rng=140.0)
    # Mastered -> the learned policy is trusted, no reflex override.
    assert dom.shield(drone, [threat], mastery_of=lambda tt: 0.9) is None


def test_simple_domain_ignores_threats_out_of_range():
    dom = SimpleDomain(mastery_threshold=0.5)
    drone = _drone(0, 0)
    far = _threat(900, 900, rng=100.0)
    assert dom.shield(drone, [far], mastery_of=lambda tt: 0.0) is None


# --- AdaptationEngine wiring --------------------------------------------

def test_engine_records_neutralization_as_engage_evidence():
    eng = AdaptationEngine(AdaptationConfig())
    events = [
        EngagementEvent(
            timestep=1, friendly_id=uuid4(), threat_id=uuid4(),
            threat_type=ThreatType.SMALL_ARMS,
            outcome=EngagementOutcome.TARGET_DESTROYED,
            friendly_health_before=1.0, friendly_health_after=1.0,
            threat_health_before=1.0, threat_health_after=0.0, distance=40.0,
        )
        for _ in range(3)
    ]
    eng.record_neutralizations(events)
    idx, conf = eng.recommend_counter(ThreatType.SMALL_ARMS)
    assert idx == _ENGAGE and conf > 0.0
    assert eng.mastery_of(ThreatType.SMALL_ARMS) > 0.0


def test_engine_record_outcome_turns_the_wheel():
    eng = AdaptationEngine(AdaptationConfig())
    for _ in range(4):
        eng.record_outcome(ThreatType.AA_MISSILE, _RETREAT, 1.0)
        eng.record_outcome(ThreatType.AA_MISSILE, _ENGAGE, -1.0)
    assert eng.recommend_counter(ThreatType.AA_MISSILE)[0] == _RETREAT


# --- C5: adaptation without weight overwrite -----------------------------

def _loss(threat: ThreatType, rng: float) -> LossEvent:
    return LossEvent(
        timestep=1, lost_unit_id=uuid4(), lost_unit_role="engagement",
        lost_unit_position_x=500.0, lost_unit_position_y=500.0,
        cause_of_loss=threat, engagement_range=rng,
    )


def test_learning_loop_can_adapt_without_touching_policy_weights():
    policy = TacticalPolicy.build()
    engine = AdaptationEngine(AdaptationConfig())
    cfg = LearningLoopConfig(loss_threshold=1, apply_weight_updates=False)
    loop = LearningLoop(policy, engine, cfg)

    before = {n: p.clone() for n, p in policy.transformer.named_parameters()}
    loop.on_loss(_loss(ThreatType.AA_MISSILE, 120.0))
    event = loop.maybe_adapt()

    assert event is not None, "adaptation should still run and be recorded"
    unchanged = all(
        torch.equal(p, before[n]) for n, p in policy.transformer.named_parameters()
    )
    assert unchanged, "policy weights were overwritten despite apply_weight_updates=False"


def test_learning_loop_still_applies_weights_when_enabled():
    policy = TacticalPolicy.build()
    engine = AdaptationEngine(AdaptationConfig())
    cfg = LearningLoopConfig(loss_threshold=1, apply_weight_updates=True)
    loop = LearningLoop(policy, engine, cfg)

    before = {n: p.clone() for n, p in policy.transformer.named_parameters()}
    loop.on_loss(_loss(ThreatType.AA_MISSILE, 120.0))
    loop.maybe_adapt()

    changed = any(
        not torch.equal(p, before[n]) for n, p in policy.transformer.named_parameters()
    )
    assert changed, "demo mode should still apply the adapted weights"
