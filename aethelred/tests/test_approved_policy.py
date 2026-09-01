"""Tests for the read-only verified-policy proposal boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from aethelred.deployment.approved_policy import ApprovedIntentPolicy, ApprovedPolicyError
from aethelred.deployment.release_verifier import (
    ActiveReleaseVerifier,
    ReleaseVerificationError,
    VerifiedReleaseArtifact,
)
from aethelred.runtime.audit import JsonlAuditJournal
from tests.test_operational_runtime import _runtime_inputs


def _artifact() -> VerifiedReleaseArtifact:
    return cast(VerifiedReleaseArtifact, SimpleNamespace(registration=SimpleNamespace(release_id="release-1")))


def test_approved_policy_records_only_a_bound_unexpired_proposal(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    policy = ApprovedIntentPolicy(_artifact(), "approved-policy-v1", lambda *_: proposal, journal)

    result = policy.propose(state, mission, now)

    assert result == proposal
    assert journal.read_all()[-1]["event_type"] == "approved_policy_proposed"


def test_approved_policy_rejects_an_unbound_or_expired_proposal(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    expired = replace(proposal, expires_at=now)
    policy = ApprovedIntentPolicy(_artifact(), "approved-policy-v1", lambda *_: expired, journal)

    with pytest.raises(ApprovedPolicyError, match="expired"):
        policy.propose(state, mission, now)
    assert journal.read_all() == []


def test_policy_loader_runs_only_after_release_verification(tmp_path) -> None:
    verified_path = tmp_path / "verified-policy.pt"
    verified_path.write_bytes(b"verified")
    calls: list[Path] = []

    class Verifier:
        def verify(self, *_args, **_kwargs):
            return SimpleNamespace(registration=SimpleNamespace(release_id="release-1"), model_path=verified_path)

    policy = ApprovedIntentPolicy.load(
        cast(ActiveReleaseVerifier, Verifier()),
        JsonlAuditJournal(tmp_path / "audit.jsonl"),
        tmp_path / "unverified-policy.pt",
        lambda path: calls.append(path) or (lambda *_: _runtime_inputs()[2]),
        "approved-policy-v1",
        code_revision="test",
    )

    assert calls == [verified_path]
    assert policy.artifact.model_path == verified_path


def test_policy_loader_does_not_run_after_verification_failure(tmp_path) -> None:
    called = False

    class RejectingVerifier:
        def verify(self, *_args, **_kwargs):
            raise ReleaseVerificationError("bad release")

    def loader(_: Path):
        nonlocal called
        called = True
        return lambda *_: _runtime_inputs()[2]

    with pytest.raises(ReleaseVerificationError):
        ApprovedIntentPolicy.load(
            cast(ActiveReleaseVerifier, RejectingVerifier()),
            JsonlAuditJournal(tmp_path / "audit.jsonl"),
            tmp_path / "policy.pt",
            loader,
            "approved-policy-v1",
            code_revision="test",
        )
    assert not called
