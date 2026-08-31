"""Durable, fail-closed health supervision for the operational runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.lifecycle import RuntimeLifecycleState, RuntimeLifecycleSupervisor


class RuntimeHealthError(PermissionError):
    """Raised when required runtime health evidence is absent, stale, or unhealthy."""


@dataclass(frozen=True)
class RuntimeHealthReport:
    """One timestamped health assertion from a required operational component."""

    component_id: str
    healthy: bool
    observed_at: datetime
    detail: str = ""


@dataclass
class RuntimeHealthSupervisor:
    """Require fresh healthy reports before autonomous command submission."""

    journal: JsonlAuditJournal
    lifecycle: RuntimeLifecycleSupervisor
    required_component_ids: tuple[str, ...]
    max_age: timedelta = timedelta(seconds=2)
    _reports: dict[str, RuntimeHealthReport] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        required = tuple(component.strip() for component in self.required_component_ids)
        if not required or any(not component for component in required) or len(set(required)) != len(required):
            raise ValueError("Required health component IDs must be non-empty and unique")
        if self.max_age <= timedelta():
            raise ValueError("Health report maximum age must be positive")
        self.required_component_ids = required

    def report(self, report: RuntimeHealthReport) -> None:
        """Persist one component health observation and withdraw authority on failure."""
        if report.component_id not in self.required_component_ids:
            raise RuntimeHealthError("Health report is for an unrequired component")
        if report.observed_at.tzinfo is None:
            raise RuntimeHealthError("Health report time must be timezone-aware")
        if type(report.healthy) is not bool:
            raise RuntimeHealthError("Health report status must be boolean")
        self._reports[report.component_id] = report
        self.journal.record(
            "runtime_health_observed",
            correlation_id=report.component_id,
            payload={
                "component_id": report.component_id,
                "healthy": report.healthy,
                "observed_at": report.observed_at,
                "detail": report.detail,
            },
        )
        if not report.healthy:
            self._withdraw_authority(f"{report.component_id}: {report.detail or 'unhealthy'}")

    def require_healthy(self, now: datetime | None = None) -> tuple[RuntimeHealthReport, ...]:
        """Return a complete fresh snapshot or fail closed and degrade the runtime."""
        checked_at = now or datetime.now(UTC)
        if checked_at.tzinfo is None:
            raise RuntimeHealthError("Health check time must be timezone-aware")
        reasons: list[str] = []
        reports: list[RuntimeHealthReport] = []
        for component_id in self.required_component_ids:
            report = self._reports.get(component_id)
            if report is None:
                reasons.append(f"missing {component_id}")
                continue
            age = checked_at - report.observed_at
            # Component clocks may be ahead by a small amount.  Reject only
            # material skew (the same bounded window used for stale reports).
            if age < -self.max_age or age > self.max_age:
                reasons.append(f"stale {component_id}")
                continue
            if not report.healthy:
                reasons.append(f"unhealthy {component_id}")
                continue
            reports.append(report)
        if reasons:
            reason = "; ".join(reasons)
            self.journal.record(
                "runtime_health_rejected",
                correlation_id="runtime",
                payload={"reason": reason, "checked_at": checked_at},
            )
            self._withdraw_authority(reason)
            raise RuntimeHealthError(f"Runtime health check failed: {reason}")
        return tuple(reports)

    def _withdraw_authority(self, reason: str) -> None:
        if self.lifecycle.state is RuntimeLifecycleState.ACTIVE:
            self.lifecycle.degrade(reason)
