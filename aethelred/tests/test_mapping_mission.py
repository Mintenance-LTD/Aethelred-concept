"""Mission-level evidence for mapping recovery and the operational command path."""

from datetime import timedelta
from unittest.mock import patch

import pytest

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.mapping_tasks import AssignmentError, MappingCoordinator, TaskStatus
from aethelred.simulation.mapping_mission import SCENARIOS, MappingMissionSimulation


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_three_units_complete_mapping_under_injected_faults(tmp_path, scenario):
    simulation = MappingMissionSimulation(tmp_path / scenario, scenario)
    result = simulation.run()
    assert result.completed
    assert result.completed_tasks == result.total_tasks == 6
    assert result.visited_samples == result.total_samples == 24
    assert result.duplicate_visits == 0
    for unit in simulation.units:
        events = unit.adapter.journal.read_all()
        started = {e["correlation_id"] for e in events if e["event_type"] == "command_execution_started"}
        approved = {e["payload"]["command_id"] for e in events
                    if e["event_type"] == "safety_decision" and e["payload"]["outcome"] == "authorised"}
        authenticated = {e["correlation_id"] for e in events if e["event_type"] == "intent_authenticated"}
        for event in events:
            if event["event_type"] == "mapping_vehicle_moved":
                assert event["correlation_id"] in started & approved
                assert simulation.mission.operating_area.contains(
                    type(unit.adapter.position)(event["payload"]["x"], event["payload"]["y"]))
            if event["event_type"] == "command_execution_started":
                assert event["payload"]["proposal_id"] in authenticated
    if scenario in {"unit_loss", "comms_loss"}:
        assert result.expired_assignments >= 1
        assert result.recovery_delay_seconds is not None
        moves = [e for e in simulation.units[0].adapter.journal.read_all()
                 if e["event_type"] == "mapping_vehicle_moved"]
        assert all(e["payload"]["simulated_at"] < str(simulation.started_at + timedelta(seconds=2))
                   or (scenario == "comms_loss" and
                       e["payload"]["simulated_at"] >= str(simulation.started_at + timedelta(seconds=8)))
                   for e in moves)
    if scenario == "comms_loss":
        assert result.stale_assignment_rejections == 1
    if scenario == "stale_sensor":
        assert result.rejected_intents == 5
        decisions = simulation.units[0].adapter.journal.read_all()
        assert sum(e["event_type"] == "safety_decision" and
                   e["payload"]["rule_ids"] == ["sensor_freshness"] for e in decisions) == 5
    assert "command_executed" in (simulation.output / "replay.html").read_text(encoding="utf-8")


def test_restart_preserves_progress_and_reassignment_fences_late_messages(tmp_path):
    simulation = MappingMissionSimulation(tmp_path / "run", "nominal")
    now = simulation.started_at
    from aethelred.runtime.mapping_tasks import MappingVehicle
    unit = simulation.units[0]
    simulation.coordinator.allocate((MappingVehicle("mapper-1", unit.adapter.position, .9),), now)
    task = next(t for t in simulation.coordinator.tasks if t.vehicle_id == "mapper-1")
    coordinator = simulation.coordinator
    coordinator.acknowledge(task.task_id, "mapper-1", task.assignment_version, now)
    coordinator.record_sample(task.task_id, "mapper-1", task.assignment_version, 0, now)
    coordinator.record_sample(task.task_id, "mapper-1", task.assignment_version, 0, now)
    recovered = MappingCoordinator(simulation.mission, JsonlAuditJournal(simulation.journal.path))
    current = recovered.owned(task.task_id, "mapper-1", task.assignment_version, now)
    assert current.completed_samples == 1
    assert current.status is TaskStatus.RUNNING
    later = now + timedelta(seconds=5)
    recovered.expire(later)
    with pytest.raises(AssignmentError):
        recovered.record_sample(task.task_id, "mapper-1", task.assignment_version, 1, later)
    assert next(t for t in recovered.tasks if t.task_id == task.task_id).completed_samples == 1


def test_incomplete_mission_is_reported_and_existing_evidence_is_preserved(tmp_path):
    output = tmp_path / "run"
    result = MappingMissionSimulation(output).run(max_ticks=1)
    assert not result.completed
    with pytest.raises(ValueError, match="empty output directory"):
        MappingMissionSimulation(output)


def test_grid_creation_and_progress_fail_closed_on_disk_failure(tmp_path):
    simulation = MappingMissionSimulation(tmp_path / "run", "nominal")
    journal = JsonlAuditJournal(tmp_path / "new-grid.jsonl")
    coordinator = MappingCoordinator(simulation.mission, journal)
    with patch.object(journal, "record", side_effect=OSError("disk full")), pytest.raises(OSError):
        coordinator.create_grid()
    assert not coordinator.tasks
    assert not MappingCoordinator(simulation.mission, journal).complete
    coordinator.create_grid()
    assert len(MappingCoordinator(simulation.mission, journal).tasks) == 6


def test_cancellation_revokes_ownership_without_discarding_progress(tmp_path):
    from aethelred.runtime.mapping_tasks import MappingVehicle
    simulation = MappingMissionSimulation(tmp_path / "run", "nominal")
    coordinator, now = simulation.coordinator, simulation.started_at
    unit = simulation.units[0]
    vehicle = MappingVehicle("mapper-1", unit.adapter.position, .9)
    coordinator.allocate((vehicle,), now)
    task = next(t for t in coordinator.tasks if t.vehicle_id == "mapper-1")
    coordinator.acknowledge(task.task_id, "mapper-1", task.assignment_version, now)
    coordinator.record_sample(task.task_id, "mapper-1", task.assignment_version, 0, now)
    coordinator.cancel(task.task_id, "mapper-1", task.assignment_version, now)
    with pytest.raises(AssignmentError):
        coordinator.heartbeat(task.task_id, "mapper-1", task.assignment_version, now)
    pending = next(t for t in coordinator.tasks if t.task_id == task.task_id)
    assert pending.completed_samples == 1 and pending.status is TaskStatus.PENDING
