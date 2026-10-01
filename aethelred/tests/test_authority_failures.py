"""Regression tests for post-approval mutation and failed durable transitions."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from aethelred.core.models import Vec2
from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.configuration import RuntimeConfigurationRegistry
from aethelred.runtime.operational import OperationalSafetySupervisor
from tests.test_operational_runtime import _runtime_inputs


def test_caller_vectors_cannot_mutate_approved_commands():
    mission, state, proposal, now = _runtime_inputs()
    external_target = Vec2(x=100, y=100)
    proposal = replace(proposal, target_position=external_target)
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    external_target.x = 99999
    assert result.command.target_position.x == 100
    with pytest.raises(FrozenInstanceError):
        result.command.target_position.x = 99999
    with pytest.raises(FrozenInstanceError):
        mission.operating_area.maximum.x = 99999


@pytest.mark.parametrize("transition", ["activate", "rollback"])
def test_failed_configuration_write_preserves_active_authority(tmp_path, transition):
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    registry = RuntimeConfigurationRegistry(journal)
    first = registry.register({"limit": 1}, "operator", "first")
    second = registry.register({"limit": 2}, "operator", "second")
    registry.activate(first.configuration_id, "operator", "reviewed")
    with patch.object(journal, "record", side_effect=OSError("disk full")), pytest.raises(OSError):
        getattr(registry, transition)(second.configuration_id, "operator", "reviewed")
    assert registry.active() == first
    assert RuntimeConfigurationRegistry(journal).active() == first


def test_failed_registration_does_not_leave_an_undurable_record(tmp_path):
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    registry = RuntimeConfigurationRegistry(journal)
    with patch.object(journal, "record", side_effect=OSError("disk full")), pytest.raises(OSError):
        registry.register({"limit": 1}, "operator", "reviewed")
    assert registry.register({"limit": 1}, "operator", "retry").revision == 1


def test_runtime_import_does_not_load_tactical_modules():
    source = str(Path(__file__).parents[1] / "src")
    result = subprocess.run([sys.executable, "-c",
        (f"import sys; sys.path.insert(0, {source!r}); import aethelred.runtime; "
        "assert 'aethelred.core.actions' not in sys.modules; "
        "assert 'aethelred.core.enums' not in sys.modules")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("transition", ["activate", "rollback"])
def test_failed_release_write_preserves_active_release(tmp_path, transition):
    from aethelred.deployment.release_ledger import ReleaseLedger
    from tests.test_release_ledger import _approved
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    ledger = ReleaseLedger(journal)
    first = ledger.register(_approved("baseline.pt"))
    second = ledger.register(_approved("candidate.pt"))
    ledger.activate(first.release_id, "operator", "reviewed")
    with patch.object(journal, "record", side_effect=OSError("disk full")), pytest.raises(OSError):
        getattr(ledger, transition)(second.release_id, "operator", "reviewed")
    assert ledger.active_release_id == first.release_id
    assert ReleaseLedger(journal).active_release_id == first.release_id
