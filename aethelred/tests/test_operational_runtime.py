"""Tests for the domain-neutral operational safety boundary."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from aethelred.core.models import BattlefieldState, Vec2
from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.operational import (
    AuthorisationOutcome,
    CommandArbiter,
    CommandReceipt,
    IntentProposal,
    Mission,
    MissionCapability,
    OperationalControlLoop,
    OperationalSafetySupervisor,
    WorldState,
)
from aethelred.runtime.simulator_adapter import SimulatorCommandAdapter


class _RecordingAdapter:
    def __init__(self) -> None:
        self.command_id = None

    def execute(self, command):
        self.command_id = command.command_id
        return CommandReceipt(command_id=command.command_id, accepted=True, recorded_at=datetime.now(UTC))


class _RecordingSimulation:
    def __init__(self, state: WorldState) -> None:
        self._state = state
        self.decision = None

    def get_current_state(self):
        from aethelred.core.enums import DroneRole
        from aethelred.core.models import DroneState

        return BattlefieldState(
            timestep=self._state.revision,
            friendly_units=[DroneState(role=DroneRole.RECON, position=self._state.position)],
        )

    def step_decision(self, decision):
        self.decision = decision
        return {}, 0.0, False, False, {}


def _runtime_inputs() -> tuple[Mission, WorldState, IntentProposal, datetime]:
    now = datetime.now(UTC)
    mission = Mission(
        mission_id=uuid4(),
        revision=1,
        valid_from=now - timedelta(minutes=1),
        valid_until=now + timedelta(minutes=1),
        allowed_capabilities=frozenset({MissionCapability.SURVEY, MissionCapability.HOLD}),
        assigned_vehicle_ids=frozenset({"vehicle-1"}),
    )
    state = WorldState(
        revision=4,
        observed_at=now,
        vehicle_id="vehicle-1",
        position=Vec2(x=50.0, y=50.0),
        healthy=True,
        navigation_valid=True,
    )
    proposal = IntentProposal(
        proposal_id=uuid4(),
        policy_id="deterministic-planner-v1",
        mission_id=mission.mission_id,
        mission_revision=mission.revision,
        state_revision=state.revision,
        vehicle_id=state.vehicle_id,
        capability=MissionCapability.SURVEY,
        target_position=Vec2(x=100.0, y=100.0),
        expires_at=now + timedelta(seconds=30),
    )
    return mission, state, proposal, now


def test_only_authorised_command_reaches_adapter(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    adapter = _RecordingAdapter()

    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    receipt = CommandArbiter(journal).execute(adapter, result)

    assert result.outcome is AuthorisationOutcome.AUTHORISED
    assert adapter.command_id == receipt.command_id
    events = journal.read_all()
    assert events[0]["event_type"] == "command_executed"
    assert events[0]["payload"]["accepted"]


def test_stale_or_disallowed_proposals_cannot_execute():
    mission, state, proposal, now = _runtime_inputs()
    stale = IntentProposal(**{**proposal.__dict__, "state_revision": state.revision - 1})
    result = OperationalSafetySupervisor().authorise(stale, state, mission, now)

    assert result.outcome is AuthorisationOutcome.REJECTED
    assert result.command is None
    with pytest.raises(PermissionError):
        CommandArbiter().execute(_RecordingAdapter(), result)


def test_control_loop_records_proposal_safety_and_command(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    loop = OperationalControlLoop(OperationalSafetySupervisor(), journal)

    loop.submit(proposal, state, mission, _RecordingAdapter(), now)

    assert [event["event_type"] for event in journal.read_all()] == [
        "intent_proposed",
        "safety_decision",
        "command_executed",
    ]


def test_simulator_adapter_never_emits_engage_actions():
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    simulation = _RecordingSimulation(state)

    SimulatorCommandAdapter(simulation).execute(result.command)

    from aethelred.core.enums import TacticalActionType

    assert simulation.decision is not None
    assert all(action.action_type is not TacticalActionType.ENGAGE for action in simulation.decision.actions)
