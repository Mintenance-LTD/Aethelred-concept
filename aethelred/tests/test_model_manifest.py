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
    )
    path = manifest.write(tmp_path / "policy.manifest.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "policy.pt"
    assert len(payload["model_sha256"]) == 64
    assert len(payload["evaluation_report_sha256"]) == 64


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
        )
