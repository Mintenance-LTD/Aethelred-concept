"""Tests for durable, revisioned operational mission registration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.missions import MissionRegistry, MissionRegistryError
from tests.test_operational_runtime import _runtime_inputs


def test_mission_registry_recovers_current_revision_and_rejects_stale_one(tmp_path) -> None:
    mission, _, _, _ = _runtime_inputs()
    revision_two = replace(mission, revision=2)
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    registry = MissionRegistry(journal)
    registry.register(mission, "operator@example.test", "Initial mission approval")
    registry.register(revision_two, "operator@example.test", "Mission revision approved")

    recovered = MissionRegistry(journal)

    assert recovered.require_registered(revision_two) == revision_two
    with pytest.raises(MissionRegistryError, match="current registered revision"):
        recovered.require_registered(mission)


def test_mission_registry_requires_operator_rationale_and_increasing_revision(tmp_path) -> None:
    mission, _, _, _ = _runtime_inputs()
    registry = MissionRegistry(JsonlAuditJournal(tmp_path / "audit.jsonl"))

    with pytest.raises(MissionRegistryError, match="identity"):
        registry.register(mission, "", "Missing identity")
    registry.register(mission, "operator@example.test", "Initial mission approval")
    with pytest.raises(MissionRegistryError, match="strictly increase"):
        registry.register(mission, "operator@example.test", "Duplicate mission revision")
