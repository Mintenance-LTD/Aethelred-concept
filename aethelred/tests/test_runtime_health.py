"""Tests for durable fail-closed runtime component health supervision."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.health import (
    ComponentHealthState,
    RuntimeHealthError,
    RuntimeHealthReport,
    RuntimeHealthSupervisor,
)
from aethelred.runtime.lifecycle import RuntimeLifecycleState, RuntimeLifecycleSupervisor
from tests.test_operational_runtime import _runtime_inputs


def _active_supervisor(tmp_path):
    mission, _, _, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    lifecycle = RuntimeLifecycleSupervisor(journal)
    lifecycle.complete_self_test(("integrity", "safety"))
    lifecycle.load_mission(mission)
    lifecycle.arm("operator@example.test")
    lifecycle.activate()
    return RuntimeHealthSupervisor(journal, lifecycle, ("estimator", "vehicle-adapter")), now


def test_health_supervisor_requires_all_fresh_healthy_components(tmp_path) -> None:
    supervisor, now = _active_supervisor(tmp_path)
    supervisor.report(RuntimeHealthReport("estimator", True, now, "state fresh"))

    with pytest.raises(RuntimeHealthError, match="missing vehicle-adapter"):
        supervisor.require_healthy(now)

    assert supervisor.lifecycle.state is RuntimeLifecycleState.DEGRADED


def test_unhealthy_or_stale_report_withdraws_runtime_authority(tmp_path) -> None:
    supervisor, now = _active_supervisor(tmp_path)
    healthy = RuntimeHealthReport("estimator", True, now, "state fresh")
    supervisor.report(healthy)
    supervisor.report(RuntimeHealthReport("vehicle-adapter", False, now, "link lost"))

    assert supervisor.lifecycle.state is RuntimeLifecycleState.DEGRADED
    with pytest.raises(RuntimeHealthError, match="unhealthy vehicle-adapter"):
        supervisor.require_healthy(now)

    restarted, restarted_now = _active_supervisor(tmp_path / "fresh")
    restarted.report(replace(healthy, observed_at=restarted_now - timedelta(seconds=3)))
    restarted.report(RuntimeHealthReport("vehicle-adapter", True, restarted_now, "link fresh"))
    with pytest.raises(RuntimeHealthError, match="stale estimator"):
        restarted.require_healthy(restarted_now)


def test_status_snapshot_is_non_mutating_and_exposes_log_and_metric_values(tmp_path) -> None:
    supervisor, now = _active_supervisor(tmp_path)
    before = supervisor.journal.read_all()

    snapshot = supervisor.status(now)

    assert not snapshot.ready
    assert snapshot.components[0].state is ComponentHealthState.MISSING
    assert snapshot.metric_values() == {
        "runtime_health_ready": 0.0,
        "runtime_health_components_required": 2.0,
        "runtime_health_components_healthy": 0.0,
        "runtime_lifecycle_active": 1.0,
    }
    assert snapshot.log_payload()["components"][0]["state"] == "missing"
    assert supervisor.journal.read_all() == before
