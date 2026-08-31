"""Tests for binding a runtime loader to the current approved release."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from aethelred.deployment.model_manifest import ModelManifest
from aethelred.deployment.promotion import HeldOutEvaluation, HumanApproval, ModelPromotionGate
from aethelred.deployment.release_ledger import ReleaseLedger
from aethelred.deployment.release_verifier import ActiveReleaseVerifier, ReleaseVerificationError
from aethelred.runtime.audit import JsonlAuditJournal


def _activate_release(tmp_path: Path, model_path: Path) -> tuple[ReleaseLedger, object]:
    report_hash = sha256((tmp_path / "report.json").read_bytes()).hexdigest()
    manifest = ModelManifest.create(
        model_path,
        tmp_path / "report.json",
        code_revision="abc123",
        configuration={"device": "cpu"},
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    )
    approved = ModelPromotionGate().approve(
        manifest,
        HeldOutEvaluation(
            candidate_id=uuid4(),
            scenario_count=20,
            candidate_metrics={"mission_success": 0.9},
            baseline_metrics={"mission_success": 0.8},
            safety_checks={"authorisation": True},
            report_sha256=report_hash,
        ),
        HumanApproval.now("approver@example.test", "Held-out review complete"),
    )
    ledger = ReleaseLedger(JsonlAuditJournal(tmp_path / "audit.jsonl"))
    registration = ledger.register(approved)
    ledger.activate(registration.release_id, "operator@example.test", "Activate reviewed release")
    return ledger, registration


def test_active_release_verifier_loads_only_matching_active_artifact(tmp_path):
    model = tmp_path / "policy.pt"
    report = tmp_path / "report.json"
    model.write_bytes(b"approved-model")
    report.write_text("evaluation evidence", encoding="utf-8")
    ledger, registration = _activate_release(tmp_path, model)
    verifier = ActiveReleaseVerifier(ledger)
    loaded_paths: list[Path] = []

    result = verifier.load(
        model,
        lambda path: loaded_paths.append(path) or "loaded-model",
        code_revision="abc123",
        configuration={"device": "cpu"},
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    )

    assert result == "loaded-model"
    assert loaded_paths == [model]
    assert verifier.verify(
        model,
        code_revision="abc123",
        configuration={"device": "cpu"},
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    ).registration == registration


def test_active_release_verifier_rejects_mismatch_before_loader_runs(tmp_path):
    model = tmp_path / "policy.pt"
    report = tmp_path / "report.json"
    model.write_bytes(b"approved-model")
    report.write_text("evaluation evidence", encoding="utf-8")
    ledger, _ = _activate_release(tmp_path, model)
    called = False

    def loader(_: Path) -> str:
        nonlocal called
        called = True
        return "must-not-load"

    with pytest.raises(ReleaseVerificationError):
        ActiveReleaseVerifier(ledger).load(
            model,
            loader,
            code_revision="wrong-revision",
            configuration={"device": "cpu"},
            observation_schema="aethelred-observation/v1",
            runtime_target="torchscript",
            training_data_reference="dataset://held-out/v1",
            runtime_environment="python=3.11;torch=2.2",
            build_provenance="build://ci/123",
        )
    assert not called


def test_active_release_verifier_requires_matching_runtime_provenance(tmp_path):
    model = tmp_path / "policy.pt"
    report = tmp_path / "report.json"
    model.write_bytes(b"approved-model")
    report.write_text("evaluation evidence", encoding="utf-8")
    ledger, _ = _activate_release(tmp_path, model)

    with pytest.raises(ReleaseVerificationError):
        ActiveReleaseVerifier(ledger).verify(
            model,
            code_revision="abc123",
            configuration={"device": "cpu"},
            observation_schema="aethelred-observation/v1",
            runtime_target="torchscript",
            training_data_reference="dataset://held-out/v1",
            runtime_environment="python=3.11;torch=2.2",
            build_provenance="build://ci/unapproved",
        )
