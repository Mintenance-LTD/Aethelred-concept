"""Tests for immutable model-release provenance manifests."""

from __future__ import annotations

import json

import pytest

from aethelred.deployment.model_manifest import ModelManifest


def test_manifest_hashes_artifact_report_and_canonical_config(tmp_path):
    model = tmp_path / "policy.pt"
    report = tmp_path / "evaluation.json"
    model.write_bytes(b"approved-model")
    report.write_text('{"passed": true}', encoding="utf-8")

    manifest = ModelManifest.create(
        model,
        report,
        code_revision="abc123",
        configuration={"target_return": 10.0, "device": "cpu"},
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    )
    path = manifest.write(tmp_path / "policy.manifest.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "policy.pt"
    assert len(payload["model_sha256"]) == 64
    assert len(payload["evaluation_report_sha256"]) == 64
    assert payload["training_data_reference"] == "dataset://held-out/v1"
    assert payload["runtime_environment"] == "python=3.11;torch=2.2"
    assert payload["build_provenance"] == "build://ci/123"


def test_manifest_requires_existing_evaluation_evidence(tmp_path):
    model = tmp_path / "policy.pt"
    model.write_bytes(b"approved-model")

    with pytest.raises(FileNotFoundError):
        ModelManifest.create(
            model,
            tmp_path / "missing.json",
            code_revision="abc123",
            configuration={},
            observation_schema="aethelred-observation/v1",
            runtime_target="torchscript",
            training_data_reference="dataset://held-out/v1",
            runtime_environment="python=3.11;torch=2.2",
            build_provenance="build://ci/123",
        )


def test_manifest_verification_rejects_artifact_or_runtime_drift(tmp_path):
    model = tmp_path / "policy.pt"
    report = tmp_path / "evaluation.json"
    model.write_bytes(b"approved-model")
    report.write_text('{"passed": true}', encoding="utf-8")
    config = {"device": "cpu", "target_return": 10.0}
    manifest = ModelManifest.create(
        model,
        report,
        code_revision="abc123",
        configuration=config,
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    )

    assert manifest.verify_artifact(
        model,
        code_revision="abc123",
        configuration={"target_return": 10.0, "device": "cpu"},
        observation_schema="aethelred-observation/v1",
        runtime_target="torchscript",
        training_data_reference="dataset://held-out/v1",
        runtime_environment="python=3.11;torch=2.2",
        build_provenance="build://ci/123",
    ) == model
    model.write_bytes(b"substituted-model")
    with pytest.raises(ValueError, match="digest"):
        manifest.verify_artifact(
            model,
            code_revision="abc123",
            configuration=config,
            observation_schema="aethelred-observation/v1",
            runtime_target="torchscript",
            training_data_reference="dataset://held-out/v1",
            runtime_environment="python=3.11;torch=2.2",
            build_provenance="build://ci/123",
        )


def test_release_provenance_rejects_legacy_manifest():
    manifest = ModelManifest(
        schema_version="1.0",
        model_name="legacy.pt",
        model_sha256="a" * 64,
        code_revision="abc123",
        configuration_sha256="b" * 64,
        observation_schema="aethelred-observation/v1",
        evaluation_report_sha256="c" * 64,
        runtime_target="torchscript",
    )

    with pytest.raises(ValueError, match="schema 1.1"):
        manifest.require_complete_provenance()
