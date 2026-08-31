"""Typed, non-offensive runtime contracts and safety authorisation.

This module is intentionally independent of the tactical simulation ontology.
It defines the production-facing control boundary: autonomy may propose a
bounded mission intent, while the safety supervisor alone can issue an
executable command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Protocol
from uuid import UUID, uuid4

from aethelred.core.models import Vec2
from aethelred.runtime.audit import JsonlAuditJournal


class MissionCapability(str, Enum):
    """Capabilities deliberately allowed in the operational runtime."""

    HOLD = "hold"
    RETURN_HOME = "return_home"
    SURVEY = "survey"
    INSPECT = "inspect"
    MAP = "map"
    RELAY = "relay"
    SEARCH = "search"


class AuthorisationOutcome(str, Enum):
    """The outcome of deterministic mission and safety validation."""

    AUTHORISED = "authorised"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OperatingArea:
    """Closed two-dimensional area an approved mission may occupy."""

    minimum: Vec2
    maximum: Vec2

    def __post_init__(self) -> None:
        values = (self.minimum.x, self.minimum.y, self.maximum.x, self.maximum.y)
        if not all(isfinite(value) for value in values):
            raise ValueError("Operating-area bounds must be finite")
        if self.minimum.x > self.maximum.x or self.minimum.y > self.maximum.y:
            raise ValueError("Operating-area minimum must not exceed maximum")

    def contains(self, position: Vec2) -> bool:
        """Return whether a finite position lies inside or on the mission boundary."""
        if not isfinite(position.x) or not isfinite(position.y):
            return False
        return (
            self.minimum.x <= position.x <= self.maximum.x
            and self.minimum.y <= position.y <= self.maximum.y
        )


@dataclass(frozen=True)
class Mission:
    """An operator-approved bounded mission revision."""

    mission_id: UUID
    revision: int
    valid_from: datetime
    valid_until: datetime
    allowed_capabilities: frozenset[MissionCapability]
    assigned_vehicle_ids: frozenset[str]
    operating_area: OperatingArea
    authorised_issuer_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("Mission revision must be positive")
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("Mission validity timestamps must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("Mission valid_until must be after valid_from")
        if not self.allowed_capabilities or not self.assigned_vehicle_ids:
            raise ValueError("Mission must grant capabilities and assigned vehicles")
        if not self.authorised_issuer_ids or any(not issuer.strip() for issuer in self.authorised_issuer_ids):
            raise ValueError("Mission must declare at least one non-empty authorised issuer")


@dataclass(frozen=True)
class WorldState:
    """A versioned state snapshot consumed by policy and safety logic."""

    revision: int
    observed_at: datetime
    vehicle_id: str
    position: Vec2
    healthy: bool
    navigation_valid: bool
    battery_reserve: float
    localisation_quality: float
    sensor_observed_at: datetime
    communications_healthy: bool
    operator_link_active: bool
    runtime_healthy: bool


@dataclass(frozen=True)
class IntentProposal:
    """A policy proposal; it has no authority to execute by itself."""

    proposal_id: UUID
    policy_id: str
    mission_id: UUID
    mission_revision: int
    state_revision: int
    vehicle_id: str
    capability: MissionCapability
    target_position: Vec2 | None
    expires_at: datetime


@dataclass(frozen=True)
class AuthorisedCommand:
    """A command issued only after a positive safety decision."""

    command_id: UUID
    proposal_id: UUID
    mission_id: UUID
    mission_revision: int
    vehicle_id: str
    capability: MissionCapability
    target_position: Vec2 | None
    expires_at: datetime
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class AuthorisationResult:
    """A complete, auditable result of safety authorisation."""

    outcome: AuthorisationOutcome
    command: AuthorisedCommand | None
    rule_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CommandReceipt:
    """An adapter acknowledgement for an authorised command."""

    command_id: UUID
    accepted: bool
    recorded_at: datetime
    detail: str = ""


class AuthorisedCommandExecutor(Protocol):
    """An adapter can execute only a safety-authorised command."""

    def execute(self, command: AuthorisedCommand) -> CommandReceipt: ...


@dataclass
class OperationalSafetySupervisor:
    """Deterministically validates mission, state, and command freshness."""

    max_state_age: timedelta = field(default=timedelta(seconds=2))
    max_sensor_age: timedelta = field(default=timedelta(seconds=1))
    min_battery_reserve: float = 0.20
    min_localisation_quality: float = 0.75

    def authorise(
        self,
        proposal: IntentProposal,
        state: WorldState,
        mission: Mission,
        now: datetime | None = None,
    ) -> AuthorisationResult:
        """Return an executable command only when all safety rules pass."""
        checked_at = now or datetime.now(UTC)
        rule_ids: list[str] = []

        timestamps = (
            checked_at,
            mission.valid_from,
            mission.valid_until,
            state.observed_at,
            state.sensor_observed_at,
            proposal.expires_at,
        )
        if any(timestamp.tzinfo is None for timestamp in timestamps):
            return self._reject("timestamp_timezone", "Safety timestamps must be timezone-aware")
        if state.observed_at > checked_at:
            return self._reject("state_timestamp", "World state observation is in the future")
        if state.sensor_observed_at > checked_at:
            return self._reject("sensor_timestamp", "Sensor observation is in the future")
        telemetry_values = (state.battery_reserve, state.localisation_quality)
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in telemetry_values):
            return self._reject("telemetry_values", "Operational telemetry values must be finite ratios")
        if proposal.mission_id != mission.mission_id or proposal.mission_revision != mission.revision:
            return self._reject("mission_identity", "Proposal does not match the approved mission")
        if proposal.vehicle_id != state.vehicle_id or proposal.vehicle_id not in mission.assigned_vehicle_ids:
            return self._reject("vehicle_assignment", "Vehicle is not assigned to this mission")
        if not mission.operating_area.contains(state.position):
            return self._reject("state_operating_area", "Vehicle state is outside the approved area")
        if not mission.valid_from <= checked_at <= mission.valid_until:
            return self._reject("mission_validity", "Mission is not currently valid")
        if proposal.capability not in mission.allowed_capabilities:
            return self._reject("capability_allowlist", "Requested capability is not allowed")
        if proposal.target_position is not None and not mission.operating_area.contains(
            proposal.target_position
        ):
            return self._reject("target_operating_area", "Target is outside the approved area")
        if proposal.state_revision != state.revision:
            return self._reject("state_revision", "Proposal was made from stale world state")
        if checked_at - state.observed_at > self.max_state_age:
            return self._reject("state_freshness", "World state has expired")
        if checked_at - state.sensor_observed_at > self.max_sensor_age:
            return self._reject("sensor_freshness", "Sensor data has expired")
        if checked_at >= proposal.expires_at:
            return self._reject("command_expiry", "Proposal has expired")
        if not state.healthy or not state.navigation_valid or not state.runtime_healthy:
            return self._reject("runtime_health", "Vehicle, navigation, or runtime health is invalid")
        if state.localisation_quality < self.min_localisation_quality:
            return self._reject("localisation_quality", "Localisation quality is below the mission threshold")
        if state.battery_reserve < self.min_battery_reserve and proposal.capability not in {
            MissionCapability.HOLD,
            MissionCapability.RETURN_HOME,
        }:
            return self._reject("battery_reserve", "Battery reserve permits only hold or return-home")
        if (
            not state.communications_healthy or not state.operator_link_active
        ) and proposal.capability not in {MissionCapability.HOLD, MissionCapability.RETURN_HOME}:
            return self._reject(
                "operator_link",
                "Communications or operator link permits only hold or return-home",
            )

        rule_ids.append("authorised")
        command = AuthorisedCommand(
            command_id=uuid4(),
            proposal_id=proposal.proposal_id,
            mission_id=mission.mission_id,
            mission_revision=mission.revision,
            vehicle_id=proposal.vehicle_id,
            capability=proposal.capability,
            target_position=proposal.target_position,
            expires_at=proposal.expires_at,
            rule_ids=tuple(rule_ids),
        )
        return AuthorisationResult(
            outcome=AuthorisationOutcome.AUTHORISED,
            command=command,
            rule_ids=tuple(rule_ids),
            reason="All mission and safety rules passed",
        )

    @staticmethod
    def _reject(rule_id: str, reason: str) -> AuthorisationResult:
        return AuthorisationResult(
            outcome=AuthorisationOutcome.REJECTED,
            command=None,
            rule_ids=(rule_id,),
            reason=reason,
        )


class CommandArbiter:
    """The sole execution boundary for an authorised operational command."""

    def __init__(self, journal: JsonlAuditJournal | None = None) -> None:
        self._journal = journal
        self._consumed_command_ids: set[UUID] = set()
        if journal is not None:
            self._recover_consumed_commands()

    def execute(
        self,
        executor: AuthorisedCommandExecutor,
        result: AuthorisationResult,
        now: datetime | None = None,
    ) -> CommandReceipt:
        """Execute one fresh authorised command and verify its acknowledgement.

        Command identifiers are consumed before invoking the adapter, including
        after process restart via journal replay. This fails closed when an
        adapter's outcome is uncertain instead of reissuing the same command.
        """
        if result.outcome is not AuthorisationOutcome.AUTHORISED or result.command is None:
            self._record_rejection("unavailable", result.reason, result.rule_ids)
            raise PermissionError("An authorised command is required for execution")

        command = result.command
        checked_at = now or datetime.now(UTC)
        if checked_at.tzinfo is None:
            self._record_rejection(str(command.command_id), "Execution time must be timezone-aware")
            raise PermissionError("Execution time must be timezone-aware")
        if command.expires_at.tzinfo is None or checked_at >= command.expires_at:
            self._record_rejection(str(command.command_id), "Authorised command has expired")
            raise PermissionError("Authorised command has expired")
        if command.command_id in self._consumed_command_ids:
            self._record_rejection(str(command.command_id), "Authorised command has already been consumed")
            raise PermissionError("Authorised command has already been consumed")

        # Consume before adapter invocation: a timeout or exception cannot be
        # safely distinguished from a partially applied external command.
        self._consumed_command_ids.add(command.command_id)
        if self._journal is not None:
            self._journal.record(
                "command_execution_started",
                correlation_id=str(command.command_id),
                payload={
                    "proposal_id": str(command.proposal_id),
                    "mission_id": str(command.mission_id),
                    "vehicle_id": command.vehicle_id,
                    "capability": command.capability.value,
                },
            )
        try:
            receipt = executor.execute(command)
        except Exception as error:
            self._record_rejection(str(command.command_id), f"Adapter execution failed: {error}")
            raise
        if receipt.command_id != command.command_id:
            self._record_rejection(str(command.command_id), "Adapter receipt command ID does not match")
            raise RuntimeError("Adapter receipt command ID does not match authorised command")
        if self._journal is not None:
            self._journal.record(
                "command_executed",
                correlation_id=str(command.command_id),
                payload={
                    "proposal_id": str(command.proposal_id),
                    "mission_id": str(command.mission_id),
                    "vehicle_id": command.vehicle_id,
                    "capability": command.capability.value,
                    "accepted": receipt.accepted,
                    "detail": receipt.detail,
                },
            )
        return receipt

    def _recover_consumed_commands(self) -> None:
        """Restore one-time-use command IDs from a verified audit journal."""
        assert self._journal is not None
        for event in self._journal.read_all():
            event_type = event.get("event_type")
            if event_type not in {"command_execution_started", "command_executed"}:
                continue
            try:
                command_id = UUID(str(event["correlation_id"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid executed-command audit record") from error
            if event_type == "command_execution_started":
                if command_id in self._consumed_command_ids:
                    raise ValueError("Duplicate command-execution start audit record")
                self._consumed_command_ids.add(command_id)
            elif command_id not in self._consumed_command_ids:
                # Support journals written before execution-start events were
                # introduced, while still treating their completed commands as
                # permanently consumed.
                self._consumed_command_ids.add(command_id)

    def _record_rejection(
        self,
        correlation_id: str,
        reason: str,
        rule_ids: tuple[str, ...] = (),
    ) -> None:
        if self._journal is not None:
            self._journal.record(
                "command_rejected",
                correlation_id=correlation_id,
                payload={"reason": reason, "rule_ids": rule_ids},
            )


class OperationalControlLoop:
    """Records only verified proposal-to-command provenance.

    Raw policy proposals are deliberately rejected by the public method. The
    authenticated wrapper below is the production entry point and invokes the
    internal verified path only after it validates the signed envelope.
    """

    def __init__(
        self,
        safety_supervisor: OperationalSafetySupervisor,
        journal: JsonlAuditJournal,
    ) -> None:
        self._safety_supervisor = safety_supervisor
        self._journal = journal
        self._arbiter = CommandArbiter(journal)

    def submit(
        self,
        proposal: IntentProposal,
        state: WorldState,
        mission: Mission,
        executor: AuthorisedCommandExecutor,
        now: datetime | None = None,
    ) -> CommandReceipt:
        """Reject raw proposals; use :class:`AuthenticatedOperationalControlLoop`."""
        del state, mission, executor, now
        self._journal.record(
            "unauthenticated_intent_rejected",
            correlation_id=str(proposal.proposal_id),
            payload={"policy_id": proposal.policy_id, "reason": "authenticated_envelope_required"},
        )
        raise PermissionError("Operational intents require an authenticated envelope")

    def _submit_verified(
        self,
        proposal: IntentProposal,
        state: WorldState,
        mission: Mission,
        executor: AuthorisedCommandExecutor,
        now: datetime | None = None,
    ) -> CommandReceipt:
        """Journal a proposal, safety decision, and authorised execution result."""
        correlation_id = str(proposal.proposal_id)
        self._journal.record(
            "telemetry_observed",
            correlation_id=correlation_id,
            payload={
                "vehicle_id": state.vehicle_id,
                "state_revision": state.revision,
                "observed_at": state.observed_at,
                "sensor_observed_at": state.sensor_observed_at,
                "position": {"x": state.position.x, "y": state.position.y},
                "healthy": state.healthy,
                "navigation_valid": state.navigation_valid,
                "battery_reserve": state.battery_reserve,
                "localisation_quality": state.localisation_quality,
                "communications_healthy": state.communications_healthy,
                "operator_link_active": state.operator_link_active,
                "runtime_healthy": state.runtime_healthy,
            },
        )
        self._journal.record(
            "intent_proposed",
            correlation_id=correlation_id,
            payload={
                "policy_id": proposal.policy_id,
                "mission_id": str(proposal.mission_id),
                "mission_revision": proposal.mission_revision,
                "state_revision": proposal.state_revision,
                "vehicle_id": proposal.vehicle_id,
                "capability": proposal.capability.value,
            },
        )
        result = self._safety_supervisor.authorise(proposal, state, mission, now)
        self._journal.record(
            "safety_decision",
            correlation_id=correlation_id,
            payload={
                "outcome": result.outcome.value,
                "rule_ids": result.rule_ids,
                "reason": result.reason,
                "command_id": str(result.command.command_id) if result.command else None,
            },
        )
        return self._arbiter.execute(executor, result, now)


class AuthenticatedOperationalControlLoop:
    """Production-facing loop that requires verified intent integrity first."""

    def __init__(
        self,
        control_loop: OperationalControlLoop,
        authenticator: object,
        mission_registry: object,
    ) -> None:
        self._control_loop = control_loop
        self._authenticator = authenticator
        self._mission_registry = mission_registry

    def submit(
        self,
        envelope: object,
        state: WorldState,
        mission: Mission,
        executor: AuthorisedCommandExecutor,
        now: datetime | None = None,
    ) -> CommandReceipt:
        """Verify authenticated intent before entering safety authorisation.

        Imports are deferred to avoid a runtime-contract import cycle.
        """
        from aethelred.runtime.integrity import (
            AuthenticatedIntent,
            IntegrityError,
            IntentAuthenticator,
        )
        from aethelred.runtime.missions import MissionRegistry, MissionRegistryError

        if not isinstance(self._authenticator, IntentAuthenticator):
            raise TypeError("Authenticated loop requires an IntentAuthenticator")
        if not isinstance(envelope, AuthenticatedIntent):
            raise TypeError("Authenticated loop requires an AuthenticatedIntent")
        if not isinstance(self._mission_registry, MissionRegistry):
            raise TypeError("Authenticated loop requires a MissionRegistry")
        if self._authenticator.journal is not self._control_loop._journal:
            raise TypeError("Intent authenticator and control loop must share one audit journal")
        try:
            registered_mission = self._mission_registry.require_registered(mission)
        except MissionRegistryError as error:
            self._control_loop._journal.record(
                "mission_authorisation_rejected",
                correlation_id=str(mission.mission_id),
                payload={"reason": str(error)},
            )
            raise PermissionError("Intent mission is not currently registered") from error
        proposal = self._authenticator.verify(envelope, now)
        if envelope.issuer_id not in registered_mission.authorised_issuer_ids:
            self._control_loop._journal.record(
                "intent_authentication_rejected",
                correlation_id=str(proposal.proposal_id),
                payload={"issuer_id": envelope.issuer_id, "reason": "issuer_not_authorised_for_mission"},
            )
            raise IntegrityError("Intent issuer is not authorised for this mission")
        self._control_loop._journal.record(
            "intent_authenticated",
            correlation_id=str(proposal.proposal_id),
            payload={"issuer_id": envelope.issuer_id, "nonce": envelope.nonce},
        )
        return self._control_loop._submit_verified(
            proposal, state, registered_mission, executor, now
        )
