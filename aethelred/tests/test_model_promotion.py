"""Tests for held-out evaluation and human approval gates."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from aethelred.deployment.attestation import HmacReleaseAttestor
from aethelred.deployment.model_manifest import ModelManifest
from aethelred.deployment.promotion import (
    HeldOutEvaluation,
    HumanApproval,
    ModelPromotionGate,
    PromotionError,
    PromotionPolicy,
)

_ATTESTOR = HmacReleaseAttestor("sil-attestor", b"a" * 32)


def _manifest(report_hash: str) -> ModelManifest:
    return ModelManifest(
        schema_version="1.1",
        model_name="candidate.pt",
        model_sha256="a" * 64,
        code_revision="abc123",
        configuration_sha256="b" * 64,
        observation_schema="aethelred-observation/v1",
        evaluation_report_sha256=report_hash,
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
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
    manifest = _manifest(evaluation.report_sha256)
    approval = HumanApproval.now("reviewer@example.test", "Held-out metrics and safety checks reviewed")
    release = ModelPromotionGate().approve(
        manifest,
        evaluation,
        approval,
        _ATTESTOR.attest(manifest, evaluation, approval),
        _ATTESTOR,
    )

    assert release.evaluation == evaluation
    assert release.approval.approver == "reviewer@example.test"
    assert release.attestation.issuer_id == "sil-attestor"


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


def test_gate_requires_declared_operational_scenario_coverage() -> None:
    evaluation = _evaluation(scenario_categories=("survey",))
    gate = ModelPromotionGate(
        PromotionPolicy(required_scenario_categories=("survey", "degraded_gps"))
    )

    with pytest.raises(PromotionError, match="degraded_gps"):
        gate.approve(
            _manifest(evaluation.report_sha256),
            evaluation,
            HumanApproval.now("reviewer@example.test", "Reviewed"),
        )


def test_gate_rejects_a_tampered_release_attestation() -> None:
    evaluation = _evaluation()
    manifest = _manifest(evaluation.report_sha256)
    approval = HumanApproval.now("reviewer@example.test", "Reviewed")
    attestation = replace(_ATTESTOR.attest(manifest, evaluation, approval), signature="b" * 64)

    with pytest.raises(PromotionError, match="attestation"):
        ModelPromotionGate().approve(manifest, evaluation, approval, attestation, _ATTESTOR)
