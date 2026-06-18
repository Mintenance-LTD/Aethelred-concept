"""Tests for the adaptation engine.

Pins the audit finding that adaptation used a constant target (always EVADE) and
never activated EWC / the replay buffer.
"""

from __future__ import annotations

from uuid import uuid4

from aethelred.adaptation.adaptation_engine import AdaptationEngine
from aethelred.config.settings import AdaptationConfig
from aethelred.core.enums import TacticalActionType, ThreatType
from aethelred.core.events import LossEvent
from aethelred.tactical_ai.policy import TacticalPolicy

_ACTION_INDEX = {a: i for i, a in enumerate(TacticalActionType)}


def _loss(threat: ThreatType, rng: float) -> LossEvent:
    return LossEvent(
        timestep=1,
        lost_unit_id=uuid4(),
        lost_unit_role="engagement",
        lost_unit_position_x=500.0,
        lost_unit_position_y=500.0,
        cause_of_loss=threat,
        killing_threat_position_x=400.0,
        killing_threat_position_y=400.0,
        engagement_range=rng,
    )


def test_counter_action_is_threat_aware():
    eng = AdaptationEngine(AdaptationConfig())
    assert eng._counter_action_index(_loss(ThreatType.EW_JAMMING, 50)) == _ACTION_INDEX[TacticalActionType.FORMATION_CHANGE]
    assert eng._counter_action_index(_loss(ThreatType.SMALL_ARMS, 10)) == _ACTION_INDEX[TacticalActionType.RETREAT]
    assert eng._counter_action_index(_loss(ThreatType.AA_MISSILE, 120)) == _ACTION_INDEX[TacticalActionType.EVADE]


def test_training_targets_are_not_constant():
    eng = AdaptationEngine(AdaptationConfig())
    data = eng._losses_to_training_data(
        [_loss(ThreatType.EW_JAMMING, 50), _loss(ThreatType.SMALL_ARMS, 10)]
    )
    targets = {int(t.item()) for _, t in data}
    assert len(targets) > 1, "adaptation still uses a single constant target"


def test_adapt_activates_ewc_and_replay():
    policy = TacticalPolicy.build()
    eng = AdaptationEngine(AdaptationConfig())
    result = eng.adapt(
        policy.transformer,
        [_loss(ThreatType.EW_JAMMING, 50), _loss(ThreatType.EW_JAMMING, 55)],
    )
    assert result.new_threat_registered
    assert eng.ewc.num_tasks >= 1, "EWC never registered a task"
    assert len(eng.replay_buffer) >= 2, "replay buffer never populated"
