"""Tests for audited approval lifecycle and rollback records."""

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
from aethelred.deployment.release_ledger import ReleaseLedger
from aethelred.runtime.audit import JsonlAuditJournal


def _approved(model_name: str):
    report_hash = ("a" if model_name == "baseline.pt" else "b") * 64
    manifest = ModelManifest(
        schema_version="1.1",
        model_name=model_name,
        model_sha256="c" * 64,
        code_revision="abc123",
        configuration_sha256="d" * 64,
        observation_schema="aethelred-observation/v1",
        evaluation_report_sha256=report_hash,
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    )
    evaluation = HeldOutEvaluation(
        candidate_id=uuid4(),
        scenario_count=20,
        candidate_metrics={"mission_success": 0.9},
        baseline_metrics={"mission_success": 0.8},
        safety_checks={"authorisation": True},
        report_sha256=report_hash,
    )
    return ModelPromotionGate().approve(
        manifest,
        evaluation,
        HumanApproval.now("approver@example.test", "Held-out review complete"),
    )


def test_release_lifecycle_is_durable_and_rollback_is_accountable(tmp_path) -> None:
    journal = JsonlAuditJournal(tmp_path / "release-audit.jsonl")
    ledger = ReleaseLedger(journal)
    baseline = ledger.register(_approved("baseline.pt"))
    candidate = ledger.register(_approved("candidate.pt"))

    ledger.activate(baseline.release_id, "operator@example.test", "Initial approved release")
    ledger.activate(candidate.release_id, "operator@example.test", "Validated improvement")
    recovered = ReleaseLedger(journal)
    assert recovered.active_release_id == candidate.release_id
    rollback = recovered.rollback(baseline.release_id, "operator@example.test", "Regression detected")

    assert rollback.from_release_id == candidate.release_id
    assert recovered.active_release_id == baseline.release_id
    assert [event["event_type"] for event in journal.read_all()] == [
        "release_registered",
        "release_registered",
        "release_activated",
        "release_activated",
        "release_rolled_back",
    ]


def test_ledger_rejects_unapproved_or_unjustified_rollback(tmp_path) -> None:
    ledger = ReleaseLedger(JsonlAuditJournal(tmp_path / "release-audit.jsonl"))
    registered = ledger.register(_approved("baseline.pt"))

    with pytest.raises(PromotionError, match="without an active release"):
        ledger.rollback(registered.release_id, "operator@example.test", "Need rollback")

    ledger.activate(registered.release_id, "operator@example.test", "Initial approved release")
    with pytest.raises(PromotionError, match="already active"):
        ledger.rollback(registered.release_id, "operator@example.test", "Need rollback")
