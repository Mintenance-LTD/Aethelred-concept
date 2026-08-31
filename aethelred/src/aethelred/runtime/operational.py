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
class Mission:
    """An operator-approved bounded mission revision."""

    mission_id: UUID
    revision: int
    valid_from: datetime
    valid_until: datetime
    allowed_capabilities: frozenset[MissionCapability]
    assigned_vehicle_ids: frozenset[str]


@dataclass(frozen=True)
class WorldState:
    """A versioned state snapshot consumed by policy and safety logic."""

    revision: int
    observed_at: datetime
    vehicle_id: str
    position: Vec2
    healthy: bool
    navigation_valid: bool


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

        if proposal.mission_id != mission.mission_id or proposal.mission_revision != mission.revision:
            return self._reject("mission_identity", "Proposal does not match the approved mission")
        if proposal.vehicle_id != state.vehicle_id or proposal.vehicle_id not in mission.assigned_vehicle_ids:
            return self._reject("vehicle_assignment", "Vehicle is not assigned to this mission")
        if not mission.valid_from <= checked_at <= mission.valid_until:
            return self._reject("mission_validity", "Mission is not currently valid")
        if proposal.capability not in mission.allowed_capabilities:
            return self._reject("capability_allowlist", "Requested capability is not allowed")
        if proposal.state_revision != state.revision:
            return self._reject("state_revision", "Proposal was made from stale world state")
        if checked_at - state.observed_at > self.max_state_age:
            return self._reject("state_freshness", "World state has expired")
        if checked_at >= proposal.expires_at:
            return self._reject("command_expiry", "Proposal has expired")
        if not state.healthy or not state.navigation_valid:
            return self._reject("vehicle_health", "Vehicle health or navigation is invalid")

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

    def execute(
        self,
        executor: AuthorisedCommandExecutor,
        result: AuthorisationResult,
    ) -> CommandReceipt:
        """Reject unauthorised proposals before they reach an adapter."""
        if result.outcome is not AuthorisationOutcome.AUTHORISED or result.command is None:
            if self._journal is not None:
                self._journal.record(
                    "command_rejected",
                    correlation_id="unavailable",
                    payload={"reason": result.reason, "rule_ids": result.rule_ids},
                )
            raise PermissionError("An authorised command is required for execution")
        receipt = executor.execute(result.command)
        if self._journal is not None:
            self._journal.record(
                "command_executed",
                correlation_id=str(result.command.command_id),
                payload={
                    "proposal_id": str(result.command.proposal_id),
                    "mission_id": str(result.command.mission_id),
                    "vehicle_id": result.command.vehicle_id,
                    "capability": result.command.capability.value,
                    "accepted": receipt.accepted,
                },
            )
        return receipt


class OperationalControlLoop:
    """Records proposal-to-command provenance through the operational boundary."""

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
        """Journal a proposal, safety decision, and authorised execution result."""
        correlation_id = str(proposal.proposal_id)
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
        return self._arbiter.execute(executor, result)
