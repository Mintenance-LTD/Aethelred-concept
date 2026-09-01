"""Tests for the read-only verified-policy proposal boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from aethelred.deployment.approved_policy import ApprovedIntentPolicy, ApprovedPolicyError
from aethelred.deployment.release_verifier import VerifiedReleaseArtifact
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
