"""Durable, fail-closed health supervision for the operational runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.lifecycle import RuntimeLifecycleState, RuntimeLifecycleSupervisor


class RuntimeHealthError(PermissionError):
    """Raised when required runtime health evidence is absent, stale, or unhealthy."""


class ComponentHealthState(StrEnum):
    """A component's current operational-readiness state."""

    HEALTHY = "healthy"
    MISSING = "missing"
    STALE = "stale"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class RuntimeHealthReport:
    """One timestamped health assertion from a required operational component."""

    component_id: str
    healthy: bool
    observed_at: datetime
    detail: str = ""


@dataclass(frozen=True)
class ComponentHealthStatus:
    """A read-only assessment of one required runtime component."""

    component_id: str
    state: ComponentHealthState
    age_seconds: float | None
    detail: str


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    """A non-mutating, structured status snapshot for logs and metrics."""

    checked_at: datetime
    lifecycle_state: RuntimeLifecycleState
    components: tuple[ComponentHealthStatus, ...]

    @property
    def ready(self) -> bool:
        """Return whether every required component is fresh and healthy."""
        return all(component.state is ComponentHealthState.HEALTHY for component in self.components)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return stable, human-readable reasons that the runtime is not ready."""
        return tuple(
            f"{component.state.value} {component.component_id}"
            for component in self.components
            if component.state is not ComponentHealthState.HEALTHY
        )

    def metric_values(self) -> dict[str, float]:
        """Return low-cardinality numeric values for a metrics collector."""
        healthy_count = sum(
            component.state is ComponentHealthState.HEALTHY for component in self.components
        )
        return {
            "runtime_health_ready": float(self.ready),
            "runtime_health_components_required": float(len(self.components)),
            "runtime_health_components_healthy": float(healthy_count),
            "runtime_lifecycle_active": float(self.lifecycle_state is RuntimeLifecycleState.ACTIVE),
        }

    def log_payload(self) -> dict[str, object]:
        """Return a JSON-safe structured payload for an application logger."""
        return {
            "checked_at": self.checked_at.isoformat(),
            "lifecycle_state": self.lifecycle_state.value,
            "ready": self.ready,
            "components": [
                {
                    "component_id": component.component_id,
                    "state": component.state.value,
                    "age_seconds": component.age_seconds,
                    "detail": component.detail,
                }
                for component in self.components
            ],
        }


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
        snapshot = self.status(now)
        if not snapshot.ready:
            reason = "; ".join(snapshot.reasons)
            self.journal.record(
                "runtime_health_rejected",
                correlation_id="runtime",
                payload={"reason": reason, "checked_at": snapshot.checked_at},
            )
            self._withdraw_authority(reason)
            raise RuntimeHealthError(f"Runtime health check failed: {reason}")
        return tuple(self._reports[component_id] for component_id in self.required_component_ids)

    def status(self, now: datetime | None = None) -> RuntimeHealthSnapshot:
        """Return current health readiness without changing runtime authority or audit state."""
        checked_at = now or datetime.now(UTC)
        if checked_at.tzinfo is None:
            raise RuntimeHealthError("Health check time must be timezone-aware")
        components: list[ComponentHealthStatus] = []
        for component_id in self.required_component_ids:
            report = self._reports.get(component_id)
            if report is None:
                components.append(
                    ComponentHealthStatus(component_id, ComponentHealthState.MISSING, None, "")
                )
                continue
            age = checked_at - report.observed_at
            age_seconds = age.total_seconds()
            # Component clocks may be ahead by a small amount.  Reject only
            # material skew (the same bounded window used for stale reports).
            if age < -self.max_age or age > self.max_age:
                state = ComponentHealthState.STALE
            elif not report.healthy:
                state = ComponentHealthState.UNHEALTHY
            else:
                state = ComponentHealthState.HEALTHY
            components.append(ComponentHealthStatus(component_id, state, age_seconds, report.detail))
        return RuntimeHealthSnapshot(checked_at, self.lifecycle.state, tuple(components))

    def _withdraw_authority(self, reason: str) -> None:
        if self.lifecycle.state is RuntimeLifecycleState.ACTIVE:
            self.lifecycle.degrade(reason)
