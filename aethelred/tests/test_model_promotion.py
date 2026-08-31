"""Tests for held-out evaluation and human approval gates."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aethelred.deployment.model_manifest import ModelManifest
from aethelred.deployment.promotion import (
    HeldOutEvaluation,
    HumanApproval,
    ModelPromotionGate,
    PromotionError,
)


def _manifest(report_hash: str) -> ModelManifest:
    return ModelManifest(
        schema_version="1.0",
        model_name="candidate.pt",
        model_sha256="a" * 64,
        code_revision="abc123",
        configuration_sha256="b" * 64,
        observation_schema="aethelred-observation/v1",
        evaluation_report_sha256=report_hash,
        runtime_target="torchscript",
    )


def _evaluation(**overrides: object) -> HeldOutEvaluation:
    values: dict[str, object] = {
        "candidate_id": uuid4(),
        "scenario_count": 20,
        "candidate_metrics": {"mission_success": 0.8},
        "baseline_metrics": {"mission_success": 0.7},
        "safety_checks": {"command_authorisation": True, "geofence": True},
        "report_sha256": "c" * 64,
    }
    values.update(overrides)
    return HeldOutEvaluation(**values)  # type: ignore[arg-type]


def test_eligible_candidate_requires_evidence_and_named_approval() -> None:
    evaluation = _evaluation()
    release = ModelPromotionGate().approve(
        _manifest(evaluation.report_sha256),
        evaluation,
        HumanApproval.now("reviewer@example.test", "Held-out metrics and safety checks reviewed"),
    )

    assert release.evaluation == evaluation
    assert release.approval.approver == "reviewer@example.test"


def test_gate_rejects_failed_safety_or_non_improving_candidate() -> None:
    evaluation = _evaluation(
        candidate_metrics={"mission_success": 0.7},
        safety_checks={"geofence": False},
    )

    with pytest.raises(PromotionError, match="failed safety checks"):
        ModelPromotionGate().approve(
            _manifest(evaluation.report_sha256),
            evaluation,
            HumanApproval.now("reviewer@example.test", "Reviewed"),
        )


def test_gate_rejects_manifest_with_different_evaluation_evidence() -> None:
    evaluation = _evaluation()

    with pytest.raises(PromotionError, match="hash does not match"):
        ModelPromotionGate().approve(
            _manifest("d" * 64),
            evaluation,
            HumanApproval.now("reviewer@example.test", "Reviewed"),
        )
