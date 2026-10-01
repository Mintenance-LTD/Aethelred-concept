"""Fail-closed lifecycle supervision for the operational runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.operational import Mission


class RuntimeLifecycleError(PermissionError):
    """Raised when a runtime lifecycle transition or command state is invalid."""


class RuntimeLifecycleState(StrEnum):
    """Explicit lifecycle states for authority to submit operational intents."""

    BOOTING = "booting"
    STANDBY = "standby"
    MISSION_LOADED = "mission_loaded"
    ARMED = "armed"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SAFE_STATE = "safe_state"
    FAULT = "fault"


@dataclass
class RuntimeLifecycleSupervisor:
    """Persist explicit runtime authority transitions and fail closed after restart."""

    journal: JsonlAuditJournal
    _state: RuntimeLifecycleState = field(init=False, default=RuntimeLifecycleState.BOOTING)
    _mission_id: UUID | None = field(init=False, default=None)
    _mission_revision: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._recover()

    @property
    def state(self) -> RuntimeLifecycleState:
        """Return the current runtime authority state."""
        return self._state

    def complete_self_test(self, checks: Iterable[str]) -> None:
        """Enter standby only after explicit, non-empty self-test evidence."""
        self._require_state(RuntimeLifecycleState.BOOTING)
        recorded_checks = tuple(check.strip() for check in checks if check.strip())
        if not recorded_checks:
            self._transition(RuntimeLifecycleState.FAULT, "self_test_failed", {"checks": []})
            raise RuntimeLifecycleError("Runtime self-test requires at least one passed check")
        self._transition(RuntimeLifecycleState.STANDBY, "self_test_passed", {"checks": recorded_checks})

    def load_mission(self, mission: Mission) -> None:
        """Bind one approved mission revision while the runtime is in standby."""
        self._require_state(RuntimeLifecycleState.STANDBY)
        self._mission_id = mission.mission_id
        self._mission_revision = mission.revision
        self._transition(
            RuntimeLifecycleState.MISSION_LOADED,
            "mission_loaded",
            {"mission_id": str(mission.mission_id), "mission_revision": mission.revision},
        )

    def arm(self, operator_id: str) -> None:
        """Record an accountable operator arm action for the loaded mission."""
        self._require_state(RuntimeLifecycleState.MISSION_LOADED)
        if not operator_id.strip():
            raise RuntimeLifecycleError("Operator identity is required to arm the runtime")
        self._transition(RuntimeLifecycleState.ARMED, "runtime_armed", {"operator_id": operator_id})

    def activate(self) -> None:
        """Allow authenticated intent submission for the armed mission only."""
        self._require_state(RuntimeLifecycleState.ARMED)
        self._transition(RuntimeLifecycleState.ACTIVE, "runtime_activated", {})

    def degrade(self, reason: str) -> None:
        """Remove command authority while a recoverable fault is investigated."""
        if self._state is not RuntimeLifecycleState.ACTIVE:
            raise RuntimeLifecycleError("Only an active runtime can enter degraded state")
        self._transition(RuntimeLifecycleState.DEGRADED, "runtime_degraded", self._reason_payload(reason))

    def recover(self, checks: Iterable[str]) -> None:
        """Restore active authority only after recorded recovery checks pass."""
        self._require_state(RuntimeLifecycleState.DEGRADED)
        recorded_checks = tuple(check.strip() for check in checks if check.strip())
        if not recorded_checks:
            self._transition(RuntimeLifecycleState.SAFE_STATE, "recovery_failed", {"checks": []})
            raise RuntimeLifecycleError("Runtime recovery requires at least one passed check")
        self._transition(RuntimeLifecycleState.ACTIVE, "runtime_recovered", {"checks": recorded_checks})

    def enter_safe_state(self, reason: str) -> None:
        """Withdraw autonomous command authority on any active lifecycle path."""
        if self._state not in {
            RuntimeLifecycleState.MISSION_LOADED,
            RuntimeLifecycleState.ARMED,
            RuntimeLifecycleState.ACTIVE,
            RuntimeLifecycleState.DEGRADED,
        }:
            raise RuntimeLifecycleError("Runtime cannot enter safe state from its current state")
        self._transition(RuntimeLifecycleState.SAFE_STATE, "safe_state_entered", self._reason_payload(reason))

    def reset_to_standby(self, operator_id: str, reason: str) -> None:
        """Require an accountable reset before a safe-state runtime can be reused."""
        self._require_state(RuntimeLifecycleState.SAFE_STATE)
        if not operator_id.strip() or not reason.strip():
            raise RuntimeLifecycleError("Operator identity and reset reason are required")
        self._mission_id = None
        self._mission_revision = None
        self._transition(
            RuntimeLifecycleState.STANDBY,
            "safe_state_reset",
            {"operator_id": operator_id, "reason": reason},
        )

    def require_active(self, mission: Mission) -> None:
        """Fail closed unless this exact mission revision is explicitly active."""
        if self._state is not RuntimeLifecycleState.ACTIVE:
            raise RuntimeLifecycleError("Runtime is not active for autonomous command submission")
        if (self._mission_id, self._mission_revision) != (mission.mission_id, mission.revision):
            raise RuntimeLifecycleError("Runtime is not active for this mission revision")

    def _recover(self) -> None:
        records = [
            event
            for event in self.journal.read_all()
            if event.get("event_type") == "runtime_lifecycle_transition"
        ]
        if not records:
            return
        latest = records[-1]
        payload = latest.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeLifecycleError("Invalid lifecycle audit record")
        try:
            previous_state = RuntimeLifecycleState(str(payload["to_state"]))
            mission_id = payload.get("mission_id")
            mission_revision = payload.get("mission_revision")
            self._mission_id = UUID(str(mission_id)) if mission_id is not None else None
            self._mission_revision = int(mission_revision) if mission_revision is not None else None
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeLifecycleError("Invalid lifecycle audit record") from error
        if previous_state in {
            RuntimeLifecycleState.MISSION_LOADED,
            RuntimeLifecycleState.ARMED,
            RuntimeLifecycleState.ACTIVE,
            RuntimeLifecycleState.DEGRADED,
        }:
            self._state = RuntimeLifecycleState.SAFE_STATE
            self._transition(
                RuntimeLifecycleState.SAFE_STATE,
                "runtime_restart_safe_state",
                {"previous_state": previous_state.value},
            )
            return
        self._state = previous_state

    def _transition(
        self,
        state: RuntimeLifecycleState,
        event: str,
        detail: Mapping[str, object],
    ) -> None:
        previous_state = self._state
        self._state = state
        self.journal.record(
            "runtime_lifecycle_transition",
            correlation_id=str(self._mission_id) if self._mission_id is not None else "runtime",
            payload={
                "event": event,
                "from_state": previous_state.value,
                "to_state": state.value,
                "mission_id": str(self._mission_id) if self._mission_id is not None else None,
                "mission_revision": self._mission_revision,
                **detail,
            },
        )

    def _require_state(self, expected: RuntimeLifecycleState) -> None:
        if self._state is not expected:
            raise RuntimeLifecycleError(
                f"Runtime transition requires {expected.value}, current state is {self._state.value}"
            )

    @staticmethod
    def _reason_payload(reason: str) -> dict[str, str]:
        if not reason.strip():
            raise RuntimeLifecycleError("Lifecycle transition reason is required")
        return {"reason": reason}
