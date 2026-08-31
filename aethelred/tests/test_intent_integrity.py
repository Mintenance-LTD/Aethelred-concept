"""Tests for authenticated, expiring, replay-resistant operational intents."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.configuration import RuntimeConfigurationRegistry
from aethelred.runtime.integrity import IntegrityError, IntentAuthenticator
from aethelred.runtime.lifecycle import RuntimeLifecycleState, RuntimeLifecycleSupervisor
from aethelred.runtime.missions import MissionRegistry
from aethelred.runtime.operational import (
    AuthenticatedOperationalControlLoop,
    CommandExecutionError,
    CommandReceipt,
    OperationalControlLoop,
    OperationalSafetySupervisor,
    RuntimeIdentity,
)
from tests.test_operational_runtime import _RecordingAdapter, _runtime_inputs


class _NackAdapter:
    def execute(self, command):
        return CommandReceipt(
            command_id=command.command_id,
            accepted=False,
            recorded_at=command.expires_at,
            sequence=command.sequence,
            detail="vehicle rejected command",
        )


def _active_lifecycle(journal, mission):
    lifecycle = RuntimeLifecycleSupervisor(journal)
    lifecycle.complete_self_test(("integrity", "safety"))
    lifecycle.load_mission(mission)
    lifecycle.arm("operator@example.test")
    lifecycle.activate()
    return lifecycle


def _runtime_identity(configuration_sha256: str) -> RuntimeIdentity:
    return RuntimeIdentity(
        release_id=uuid4(),
        software_revision="aethelred-test-build",
        model_sha256="a" * 64,
        configuration_sha256=configuration_sha256,
    )


def _active_configuration(journal) -> RuntimeConfigurationRegistry:
    registry = RuntimeConfigurationRegistry(journal)
    record = registry.register(
        {"runtime": {"deadline_ms": 250}},
        "operator@example.test",
        "Reviewed runtime configuration",
    )
    registry.activate(record.configuration_id, "operator@example.test", "Activate reviewed configuration")
    return registry


def _authenticated_loop(journal, authenticator, mission):
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Approved non-offensive mission")
    configuration_registry = _active_configuration(journal)
    return AuthenticatedOperationalControlLoop(
        OperationalControlLoop(
            OperationalSafetySupervisor(), journal, _runtime_identity(configuration_registry.active().sha256)
        ),
        authenticator,
        registry,
        _active_lifecycle(journal, mission),
        configuration_registry,
    )


def test_authenticated_intent_is_verified_before_safety_and_execution(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    loop = _authenticated_loop(journal, authenticator, mission)
    adapter = _RecordingAdapter()

    loop.submit(envelope, state, mission, adapter, now)

    assert adapter.command_id is not None
    assert [event["event_type"] for event in journal.read_all()] == [
        "mission_registered",
        "runtime_configuration_registered",
        "runtime_configuration_activated",
        "runtime_lifecycle_transition",
        "runtime_lifecycle_transition",
        "runtime_lifecycle_transition",
        "runtime_lifecycle_transition",
        "intent_nonce_consumed",
        "intent_authenticated",
        "telemetry_observed",
        "intent_proposed",
        "safety_decision",
        "command_execution_started",
        "command_executed",
    ]
    attributed_events = [
        event
        for event in journal.read_all()
        if event["event_type"]
        in {
            "intent_authenticated",
            "telemetry_observed",
            "intent_proposed",
            "safety_decision",
            "command_execution_started",
            "command_executed",
        }
    ]
    for event in attributed_events:
        assert {"release_id", "software_revision", "model_sha256", "configuration_sha256"} <= set(
            event["payload"]
        )
    telemetry = next(event for event in attributed_events if event["event_type"] == "telemetry_observed")
    assert telemetry["payload"]["observation_source_id"] == "simulated-estimator"
    assert telemetry["payload"]["coordinate_frame"] == "local-enu"
    assert telemetry["payload"]["observation_sequence"] == 42


def test_tampered_or_replayed_intent_cannot_reach_adapter(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    loop = _authenticated_loop(journal, authenticator, mission)
    adapter = _RecordingAdapter()
    tampered = replace(envelope, issuer_id="untrusted-service")

    with pytest.raises(IntegrityError, match="signature"):
        loop.submit(tampered, state, mission, adapter, now)
    loop.submit(envelope, state, mission, adapter, now)
    with pytest.raises(IntegrityError, match="already been used"):
        loop.submit(envelope, state, mission, adapter, now)
    assert adapter.command_id is not None


def test_intent_authentication_rotates_keys_without_recording_key_material(tmp_path) -> None:
    _, _, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal, key_id="planner-v1")
    retained = authenticator.sign(proposal, "planner-service", issued_at=now, nonce="retained")
    retired = authenticator.sign(proposal, "planner-service", issued_at=now, nonce="retired")

    authenticator.rotate("planner-v2", b"b" * 32)
    current = authenticator.sign(proposal, "planner-service", issued_at=now, nonce="current")

    assert retained.key_id == "planner-v1"
    assert current.key_id == "planner-v2"
    authenticator.verify(retained, now)
    authenticator.verify(current, now)
    authenticator.rotate("planner-v2", b"b" * 32, retire_key_ids=("planner-v1",))
    with pytest.raises(IntegrityError, match="not trusted"):
        authenticator.verify(retired, now)
    rotation_events = [
        event for event in journal.read_all() if event["event_type"] == "intent_authentication_key_rotated"
    ]
    assert rotation_events[-1]["correlation_id"] == "planner-v2"
    assert "b" * 32 not in str(rotation_events)


def test_signed_but_unapproved_issuer_cannot_reach_adapter(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "other-planner-service", issued_at=now)
    loop = _authenticated_loop(journal, authenticator, mission)
    adapter = _RecordingAdapter()

    with pytest.raises(IntegrityError, match="not authorised"):
        loop.submit(envelope, state, mission, adapter, now)

    assert adapter.command_id is None
    assert [event["event_type"] for event in journal.read_all()][-2:] == [
        "intent_nonce_consumed",
        "intent_authentication_rejected",
    ]


def test_expired_intent_is_rejected_before_command_authorisation(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal, max_age=timedelta(seconds=1))
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now - timedelta(seconds=2))
    loop = _authenticated_loop(journal, authenticator, mission)

    with pytest.raises(IntegrityError, match="expired"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)


def test_unregistered_mission_cannot_reach_authentication_or_adapter(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    configuration_registry = _active_configuration(journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(
            OperationalSafetySupervisor(), journal, _runtime_identity(configuration_registry.active().sha256)
        ),
        authenticator,
        MissionRegistry(journal),
        _active_lifecycle(journal, mission),
        configuration_registry,
    )

    with pytest.raises(PermissionError, match="not currently registered"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)

    assert journal.read_all()[-1]["event_type"] == "mission_authorisation_rejected"


def test_nonce_replay_is_rejected_after_authenticator_restart(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    original = IntentAuthenticator(b"a" * 32, journal)
    envelope = original.sign(proposal, "planner-service", issued_at=now)
    _authenticated_loop(journal, original, mission).submit(
        envelope, state, mission, _RecordingAdapter(), now
    )

    restarted_journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    restarted = IntentAuthenticator(b"a" * 32, restarted_journal)
    lifecycle = RuntimeLifecycleSupervisor(restarted_journal)
    lifecycle.reset_to_standby("operator@example.test", "Restart recovery reviewed")
    lifecycle.load_mission(mission)
    lifecycle.arm("operator@example.test")
    lifecycle.activate()
    configuration_registry = RuntimeConfigurationRegistry(restarted_journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(
            OperationalSafetySupervisor(),
            restarted_journal,
            _runtime_identity(configuration_registry.active().sha256),
        ),
        restarted,
        MissionRegistry(restarted_journal),
        lifecycle,
        configuration_registry,
    )
    with pytest.raises(IntegrityError, match="already been used"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)


def test_authenticated_loop_rejects_an_authenticator_on_another_journal(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "runtime-audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, JsonlAuditJournal(tmp_path / "other-audit.jsonl"))
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Approved non-offensive mission")
    configuration_registry = _active_configuration(journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(
            OperationalSafetySupervisor(), journal, _runtime_identity(configuration_registry.active().sha256)
        ),
        authenticator,
        registry,
        _active_lifecycle(journal, mission),
        configuration_registry,
    )

    with pytest.raises(TypeError, match="share one audit journal"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)


def test_authenticated_loop_rejects_missing_runtime_identity(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Approved non-offensive mission")
    configuration_registry = _active_configuration(journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), journal),
        authenticator,
        registry,
        _active_lifecycle(journal, mission),
        configuration_registry,
    )

    with pytest.raises(TypeError, match="RuntimeIdentity"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)


def test_authenticated_loop_rejects_configuration_digest_mismatch(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Approved non-offensive mission")
    configuration_registry = _active_configuration(journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(OperationalSafetySupervisor(), journal, _runtime_identity("c" * 64)),
        authenticator,
        registry,
        _active_lifecycle(journal, mission),
        configuration_registry,
    )

    with pytest.raises(PermissionError, match="does not match"):
        loop.submit(envelope, state, mission, _RecordingAdapter(), now)

    assert journal.read_all()[-1]["event_type"] == "runtime_configuration_rejected"


def test_adapter_nack_transitions_active_runtime_to_safe_state(tmp_path) -> None:
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    authenticator = IntentAuthenticator(b"a" * 32, journal)
    envelope = authenticator.sign(proposal, "planner-service", issued_at=now)
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Approved non-offensive mission")
    lifecycle = _active_lifecycle(journal, mission)
    configuration_registry = _active_configuration(journal)
    loop = AuthenticatedOperationalControlLoop(
        OperationalControlLoop(
            OperationalSafetySupervisor(), journal, _runtime_identity(configuration_registry.active().sha256)
        ),
        authenticator,
        registry,
        lifecycle,
        configuration_registry,
    )

    with pytest.raises(CommandExecutionError, match="negatively acknowledged"):
        loop.submit(envelope, state, mission, _NackAdapter(), now)

    assert lifecycle.state is RuntimeLifecycleState.SAFE_STATE
    assert journal.read_all()[-1]["payload"]["event"] == "safe_state_entered"
