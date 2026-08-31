"""Tests for explicit fail-closed operational runtime lifecycle authority."""

from __future__ import annotations

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.lifecycle import (
    RuntimeLifecycleError,
    RuntimeLifecycleState,
    RuntimeLifecycleSupervisor,
)
from tests.test_operational_runtime import _runtime_inputs


def _activate(lifecycle: RuntimeLifecycleSupervisor, mission) -> None:
    lifecycle.complete_self_test(("integrity", "safety"))
    lifecycle.load_mission(mission)
    lifecycle.arm("operator@example.test")
    lifecycle.activate()


def test_lifecycle_requires_self_test_and_operator_arm_before_active(tmp_path) -> None:
    mission, _, _, _ = _runtime_inputs()
    lifecycle = RuntimeLifecycleSupervisor(JsonlAuditJournal(tmp_path / "audit.jsonl"))

    with pytest.raises(RuntimeLifecycleError, match="booting"):
        lifecycle.load_mission(mission)

    _activate(lifecycle, mission)

    assert lifecycle.state is RuntimeLifecycleState.ACTIVE
    lifecycle.require_active(mission)


def test_active_runtime_restarts_in_safe_state_and_requires_reset(tmp_path) -> None:
    mission, _, _, _ = _runtime_inputs()
    path = tmp_path / "audit.jsonl"
    _activate(RuntimeLifecycleSupervisor(JsonlAuditJournal(path)), mission)

    restarted = RuntimeLifecycleSupervisor(JsonlAuditJournal(path))

    assert restarted.state is RuntimeLifecycleState.SAFE_STATE
    with pytest.raises(RuntimeLifecycleError, match="not active"):
        restarted.require_active(mission)
    restarted.reset_to_standby("operator@example.test", "Restart verified")
    restarted.load_mission(mission)
    restarted.arm("operator@example.test")
    restarted.activate()
    restarted.require_active(mission)


def test_degraded_runtime_cannot_submit_until_recovery_checks_pass(tmp_path) -> None:
    mission, _, _, _ = _runtime_inputs()
    lifecycle = RuntimeLifecycleSupervisor(JsonlAuditJournal(tmp_path / "audit.jsonl"))
    _activate(lifecycle, mission)
    lifecycle.degrade("estimator restart")

    with pytest.raises(RuntimeLifecycleError, match="not active"):
        lifecycle.require_active(mission)
    lifecycle.recover(("estimator fresh",))

    lifecycle.require_active(mission)
