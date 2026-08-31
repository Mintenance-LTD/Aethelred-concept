"""Tests for immutable, auditable runtime configuration activation and rollback."""

from __future__ import annotations

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.configuration import RuntimeConfigurationError, RuntimeConfigurationRegistry


def test_registered_configuration_is_canonical_and_immutable(tmp_path) -> None:
    registry = RuntimeConfigurationRegistry(JsonlAuditJournal(tmp_path / "audit.jsonl"))

    record = registry.register(
        {"runtime": {"deadline_ms": 250}, "feature_enabled": True},
        "operator@example.test",
        "Reviewed runtime limits",
    )
    registry.activate(record.configuration_id, "operator@example.test", "Activate reviewed limits")

    assert registry.active().sha256 == record.sha256
    with pytest.raises(TypeError):
        registry.active().values["feature_enabled"] = False
    with pytest.raises(TypeError):
        registry.active().values["runtime"]["deadline_ms"] = 100


def test_configuration_rollback_recovers_from_the_durable_journal(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    registry = RuntimeConfigurationRegistry(JsonlAuditJournal(path))
    first = registry.register({"runtime": {"deadline_ms": 250}}, "operator@example.test", "Initial")
    second = registry.register({"runtime": {"deadline_ms": 100}}, "operator@example.test", "Tighter")
    registry.activate(first.configuration_id, "operator@example.test", "Activate initial")
    registry.activate(second.configuration_id, "operator@example.test", "Activate tighter")
    registry.rollback(first.configuration_id, "operator@example.test", "Restore known-good limits")

    recovered = RuntimeConfigurationRegistry(JsonlAuditJournal(path))

    assert recovered.active().configuration_id == first.configuration_id
    assert recovered.active().revision == 1


def test_configuration_rejects_secrets_and_unregistered_activation(tmp_path) -> None:
    registry = RuntimeConfigurationRegistry(JsonlAuditJournal(tmp_path / "audit.jsonl"))

    with pytest.raises(RuntimeConfigurationError, match="secrets"):
        registry.register({"service": {"api_key": "do-not-store"}}, "operator@example.test", "Bad")
    with pytest.raises(RuntimeConfigurationError, match="No runtime configuration"):
        registry.active()
