"""Tests for hash-chained durable audit records."""

from __future__ import annotations

import json

import pytest

from aethelred.runtime.audit import AuditIntegrityError, JsonlAuditJournal


def test_audit_journal_links_successive_events_with_hashes(tmp_path) -> None:
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    first = journal.record("intent_proposed", "proposal-1", {"source": "test"})
    second = journal.record("safety_decision", "proposal-1", {"outcome": "authorised"})

    events = journal.read_all()

    assert events[0]["event_hash"] == first.event_hash
    assert events[1]["previous_hash"] == second.previous_hash == first.event_hash


def test_audit_journal_rejects_tampered_event_content(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    journal = JsonlAuditJournal(path)
    journal.record("intent_proposed", "proposal-1", {"source": "test"})
    journal.record("safety_decision", "proposal-1", {"outcome": "authorised"})
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"]["source"] = "altered"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    with pytest.raises(AuditIntegrityError, match="line 1"):
        journal.read_all()
