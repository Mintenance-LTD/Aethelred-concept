"""Tests for hash-chained durable audit records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import pytest

from aethelred.runtime.audit import AuditIntegrityError, JsonlAuditJournal


def _record_from_separate_process(path: str, index: int) -> None:
    JsonlAuditJournal(path).record("telemetry_observed", str(index), {"index": index})


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


def test_concurrent_journal_instances_preserve_one_hash_chain(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"

    def record(index: int) -> None:
        JsonlAuditJournal(path).record("telemetry_observed", str(index), {"index": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(32)))

    events = JsonlAuditJournal(path).read_all()
    assert len(events) == 32
    assert {event["correlation_id"] for event in events} == {str(index) for index in range(32)}


def test_separate_process_writers_preserve_one_hash_chain(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    context = get_context("spawn")
    processes = [
        context.Process(target=_record_from_separate_process, args=(str(path), index))
        for index in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    events = JsonlAuditJournal(path).read_all()
    assert len(events) == 8
    assert {event["correlation_id"] for event in events} == {str(index) for index in range(8)}
