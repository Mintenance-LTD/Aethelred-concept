"""Regression tests for the offline-only adaptation boundary."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import torch

from aethelred.adaptation.adaptation_engine import AdaptationResult
from aethelred.config.settings import LearningLoopConfig
from aethelred.core.enums import ThreatType
from aethelred.core.events import LossEvent
from aethelred.learning.learning_loop import LearningLoop, OfflineAdaptationCandidate


def _loss() -> LossEvent:
    return LossEvent(
        timestep=1,
        lost_unit_id=uuid4(),
        lost_unit_role="research",
        lost_unit_position_x=1.0,
        lost_unit_position_y=1.0,
        cause_of_loss=ThreatType.EW_JAMMING,
    )


def test_adaptation_only_emits_an_offline_candidate() -> None:
    """A learning-loop adaptation must not overwrite or propagate active weights."""
    policy = MagicMock()
    policy.get_policy_weights.return_value = {"transformer.weight": torch.tensor([1.0])}
    policy.transformer = object()
    policy.state_encoder.named_parameters.return_value = []

    engine = MagicMock()
    engine.known_threat_types = set()
    engine.adapt.return_value = AdaptationResult(
        updated_weights={"weight": torch.tensor([2.0])},
        method="test",
        loss_before=1.0,
        loss_after=0.5,
        new_threat_registered=True,
    )

    candidates: list[OfflineAdaptationCandidate] = []
    loop = LearningLoop(
        tactical_policy=policy,
        adaptation_engine=engine,
        config=LearningLoopConfig(loss_threshold=1, enable_prediction=False),
        candidate_sink=candidates.append,
    )
    loop.on_loss(_loss())

    event = loop.maybe_adapt()

    assert event is not None
    policy.load_policy_weights.assert_not_called()
    engine.get_model_delta.assert_not_called()
    assert len(candidates) == 1
    assert candidates[0].adaptation_event == event
    assert candidates[0].updated_weights["transformer.weight"].item() == 2.0
