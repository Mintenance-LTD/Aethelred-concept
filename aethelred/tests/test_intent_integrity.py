"""Tests for authenticated, expiring, replay-resistant operational intents."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.integrity import IntegrityError, IntentAuthenticator
from aethelred.runtime.operational import (
    AuthenticatedOperationalControlLoop,
    OperationalControlLoop,
    OperationalSafetySupervisor,
)
from tests.test_operational_runtime import _RecordingAdapter, _runtime_inputs


def test_authenticated_intent_is_verified_before_safety_and_execution(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), journal), authenticator
    )
    adapter = _RecordingAdapter()

    loop.submit(envelope, state, mission, adapter, now)

    assert adapter.command_id is not None
    assert [event["event_type"] for event in journal.read_all()] == [
        "intent_authenticated",
        "intent_proposed",
        "safety_decision",
        "command_execution_started",
        "command_executed",
    ]


def test_tampered_or_replayed_intent_cannot_reach_adapter(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    authenticator = IntentAuthenticator(b"a" * 32)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), JsonlAuditJournal(tmp_path / "audit.jsonl")),
        authenticator,
    )
    adapter = _RecordingAdapter()
    tampered = replace(envelope, issuer_id="untrusted-service")

    with pytest.raises(IntegrityError, match="signature"):
        loop.submit(tampered, state, mission, adapter, now)
    loop.submit(envelope, state, mission, adapter, now)
    with pytest.raises(IntegrityError, match="already been used"):
        loop.submit(envelope, state, mission, adapter, now)
    assert adapter.command_id is not None


def test_signed_but_unapproved_issuer_cannot_reach_adapter(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32)
    envelope = authenticator.sign(proposal, "other-planner-service", issued_at=now)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), journal), authenticator
    )
    adapter = _RecordingAdapter()

    with pytest.raises(IntegrityError, match="not authorised"):
        loop.submit(envelope, state, mission, adapter, now)

    assert adapter.command_id is None
    assert [event["event_type"] for event in journal.read_all()] == [
        "intent_authentication_rejected"
    ]


def test_expired_intent_is_rejected_before_command_authorisation(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    authenticator = IntentAuthenticator(b"a" * 32, max_age=timedelta(seconds=1))
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now - timedelta(seconds=2))
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), JsonlAuditJournal(tmp_path / "audit.jsonl")),
        authenticator,
    )

    with pytest.raises(IntegrityError, match="expired"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)
