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
    CommandExecutionError,
    CommandReceipt,
    IntentProposal,
    Mission,
    MissionCapability,
    OperatingArea,
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


class _MismatchedReceiptAdapter:
    def execute(self, command):
        return CommandReceipt(command_id=uuid4(), accepted=True, recorded_at=datetime.now(UTC))


class _NackAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, command):
        self.calls += 1
        return CommandReceipt(
            command_id=command.command_id,
            accepted=False,
            recorded_at=datetime.now(UTC),
            detail="vehicle controller rejected command",
        )


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
        operating_area=OperatingArea(minimum=Vec2(x=0.0, y=0.0), maximum=Vec2(x=500.0, y=500.0)),
        authorised_issuer_ids=frozenset({"planner-service"}),
    )
    state = WorldState(
        revision=4,
        observed_at=now,
        vehicle_id="vehicle-1",
        position=Vec2(x=50.0, y=50.0),
        healthy=True,
        navigation_valid=True,
        battery_reserve=0.80,
        localisation_quality=0.95,
        sensor_observed_at=now,
        communications_healthy=True,
        operator_link_active=True,
        runtime_healthy=True,
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
    assert [event["event_type"] for event in events] == [
        "command_execution_started",
        "command_executed",
    ]
    assert events[1]["payload"]["accepted"]


def test_stale_or_disallowed_proposals_cannot_execute():
    mission, state, proposal, now = _runtime_inputs()
    stale = IntentProposal(**{**proposal.__dict__, "state_revision": state.revision - 1})
    result = OperationalSafetySupervisor().authorise(stale, state, mission, now)

    assert result.outcome is AuthorisationOutcome.REJECTED
    assert result.command is None
    with pytest.raises(PermissionError):
        CommandArbiter().execute(_RecordingAdapter(), result)


def test_nacked_command_is_audited_and_cannot_be_reissued(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    arbiter = CommandArbiter(journal)
    adapter = _NackAdapter()

    with pytest.raises(CommandExecutionError, match="negatively acknowledged"):
        arbiter.execute(adapter, result, now)
    with pytest.raises(PermissionError, match="already been consumed"):
        arbiter.execute(adapter, result, now)

    assert adapter.calls == 1
    assert [event["event_type"] for event in journal.read_all()] == [
        "command_execution_started",
        "command_nacked",
        "command_rejected",
    ]


def test_future_state_cannot_be_authorised():
    mission, state, proposal, now = _runtime_inputs()
    future_state = WorldState(**{**state.__dict__, "observed_at": now + timedelta(seconds=1)})

    result = OperationalSafetySupervisor().authorise(proposal, future_state, mission, now)

    assert result.outcome is AuthorisationOutcome.REJECTED
    assert result.rule_ids == ("state_timestamp",)


def test_operating_area_rejects_outside_state_or_target():
    mission, state, proposal, now = _runtime_inputs()
    outside_target = IntentProposal(**{**proposal.__dict__, "target_position": Vec2(x=501.0, y=100.0)})
    outside_state = WorldState(**{**state.__dict__, "position": Vec2(x=-1.0, y=50.0)})

    target_result = OperationalSafetySupervisor().authorise(outside_target, state, mission, now)
    state_result = OperationalSafetySupervisor().authorise(proposal, outside_state, mission, now)

    assert target_result.rule_ids == ("target_operating_area",)
    assert state_result.rule_ids == ("state_operating_area",)


def test_operating_area_rejects_invalid_bounds_and_non_finite_positions():
    with pytest.raises(ValueError, match="minimum"):
        OperatingArea(minimum=Vec2(x=2.0, y=0.0), maximum=Vec2(x=1.0, y=1.0))
    area = OperatingArea(minimum=Vec2(x=0.0, y=0.0), maximum=Vec2(x=1.0, y=1.0))
    assert not area.contains(Vec2(x=float("nan"), y=0.0))


@pytest.mark.parametrize(
    ("changes", "expected_rule"),
    [
        ({"battery_reserve": 0.19}, "battery_reserve"),
        ({"localisation_quality": 0.74}, "localisation_quality"),
        ({"sensor_observed_at": datetime.now(UTC) - timedelta(seconds=2)}, "sensor_freshness"),
        ({"communications_healthy": False}, "operator_link"),
        ({"operator_link_active": False}, "operator_link"),
        ({"runtime_healthy": False}, "runtime_health"),
        ({"battery_reserve": float("nan")}, "telemetry_values"),
    ],
)
def test_runtime_health_constraints_fail_closed(changes, expected_rule):
    mission, state, proposal, now = _runtime_inputs()
    if "sensor_observed_at" in changes:
        changes = {**changes, "sensor_observed_at": now - timedelta(seconds=2)}
    constrained_state = WorldState(**{**state.__dict__, **changes})

    result = OperationalSafetySupervisor().authorise(proposal, constrained_state, mission, now)

    assert result.outcome is AuthorisationOutcome.REJECTED
    assert result.rule_ids == (expected_rule,)


def test_low_reserve_and_lost_link_allow_return_home_or_hold():
    mission, state, proposal, now = _runtime_inputs()
    constrained_state = WorldState(
        **{
            **state.__dict__,
            "battery_reserve": 0.19,
            "communications_healthy": False,
            "operator_link_active": False,
        }
    )
    return_proposal = IntentProposal(
        **{**proposal.__dict__, "capability": MissionCapability.RETURN_HOME}
    )
    permitted_mission = Mission(
        **{
            **mission.__dict__,
            "allowed_capabilities": frozenset({MissionCapability.RETURN_HOME}),
        }
    )

    result = OperationalSafetySupervisor().authorise(
        return_proposal, constrained_state, permitted_mission, now
    )

    assert result.outcome is AuthorisationOutcome.AUTHORISED


def test_command_expiry_replay_and_restart_are_rejected(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    adapter = _RecordingAdapter()

    arbiter = CommandArbiter(journal)
    arbiter.execute(adapter, result, now)
    with pytest.raises(PermissionError, match="already been consumed"):
        arbiter.execute(adapter, result, now)
    with pytest.raises(PermissionError, match="already been consumed"):
        CommandArbiter(journal).execute(adapter, result, now)

    events = journal.read_all()
    assert [event["event_type"] for event in events] == [
        "command_execution_started",
        "command_executed",
        "command_rejected",
        "command_rejected",
    ]


def test_expired_command_and_mismatched_receipt_fail_closed(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")

    with pytest.raises(PermissionError, match="has expired"):
        CommandArbiter(journal).execute(_RecordingAdapter(), result, proposal.expires_at)

    fresh_result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    with pytest.raises(RuntimeError, match="does not match"):
        CommandArbiter(journal).execute(_MismatchedReceiptAdapter(), fresh_result, now)
    with pytest.raises(PermissionError, match="already been consumed"):
        CommandArbiter(journal).execute(_RecordingAdapter(), fresh_result, now)


def test_raw_control_loop_rejects_unauthenticated_proposal(tmp_path):
    mission, state, proposal, now = _runtime_inputs()
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    loop = OperationalControlLoop(OperationalSafetySupervisor(), journal)

    with pytest.raises(PermissionError, match="authenticated envelope"):
        loop.submit(proposal, state, mission, _RecordingAdapter(), now)

    assert [event["event_type"] for event in journal.read_all()] == [
        "unauthenticated_intent_rejected",
    ]


def test_simulator_adapter_never_emits_engage_actions():
    mission, state, proposal, now = _runtime_inputs()
    result = OperationalSafetySupervisor().authorise(proposal, state, mission, now)
    simulation = _RecordingSimulation(state)

    SimulatorCommandAdapter(simulation).execute(result.command)

    from aethelred.core.enums import TacticalActionType

    assert simulation.decision is not None
    assert all(action.action_type is not TacticalActionType.ENGAGE for action in simulation.decision.actions)
