"""Evidence for offline training and bounded allocator proposals."""

import json
import runpy
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from aethelred.learning.mapping_allocator import (
    BalancedTaskAllocator,
    DurationModel,
    train_duration_model,
)
from aethelred.runtime.mapping_allocation import TaskAssignment
from aethelred.runtime.mapping_tasks import AssignmentError, MappingVehicle, TaskStatus
from aethelred.simulation.mapping_mission import MappingLayout, MappingMissionSimulation


def test_offline_model_is_reproducible_and_immutable(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    report = train_duration_model(first, (1, 2, 3), (501, 502))
    train_duration_model(second, (1, 2, 3), (501, 502))
    model = DurationModel.load(first)
    assert model.artifact_sha256 == DurationModel.load(second).artifact_sha256
    assert set(report["training_seeds"]).isdisjoint(report["validation_seeds"])
    assert report["validation_mae_ticks"] < 1.0
    with pytest.raises(FrozenInstanceError):
        model.coefficients = (0., 0., 0., 0.)
    with pytest.raises(ValueError, match="disjoint"):
        train_duration_model(tmp_path / "overlap.json", (1, 2), (2, 3))


def test_nonfinite_weights_cannot_be_loaded(tmp_path):
    path = tmp_path / "model.json"
    train_duration_model(path, (1,), (501,))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["coefficients"][0] = float("nan")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        DurationModel.load(path)


@pytest.mark.parametrize("mode", ["unknown_task", "unknown_vehicle", "duplicate_task", "duplicate_vehicle"])
def test_allocator_cannot_assign_unknown_or_duplicate_resources(tmp_path, mode):
    class InvalidAllocator:
        allocator_id = "invalid-test"
        artifact_sha256 = None

        def propose(self, vehicles, tasks):
            first = TaskAssignment(vehicles[0].vehicle_id, tasks[0].task_id)
            if mode == "unknown_task":
                return (first, TaskAssignment(vehicles[1].vehicle_id, uuid4()))
            if mode == "unknown_vehicle":
                return (first, TaskAssignment("unregistered", tasks[1].task_id))
            if mode == "duplicate_task":
                return (first, TaskAssignment(vehicles[1].vehicle_id, tasks[0].task_id))
            return (first, TaskAssignment(vehicles[0].vehicle_id, tasks[1].task_id))

    simulation = MappingMissionSimulation(tmp_path / mode, allocator=InvalidAllocator())
    with pytest.raises(AssignmentError):
        simulation.coordinator.allocate(tuple(MappingVehicle(u.adapter.vehicle_id, u.adapter.position, .9)
                                               for u in simulation.units), simulation.started_at)
    assert all(t.status is TaskStatus.PENDING for t in simulation.coordinator.tasks)
    assert simulation.journal.read_all()[-1]["event_type"] == "mapping_allocation_rejected"


def test_learned_allocator_survives_coordinator_restart_and_keeps_safety_path(tmp_path):
    path = tmp_path / "model.json"
    train_duration_model(path, (1, 2), (501,))
    model = DurationModel.load(path)
    simulation = MappingMissionSimulation(tmp_path / "run", "coordinator_restart",
                                          layout=MappingLayout.from_seed(9000),
                                          allocator=BalancedTaskAllocator(model))
    result = simulation.run()
    assert result.completed and result.duplicate_visits == 0
    assert simulation.coordinator.allocator.artifact_sha256 == model.artifact_sha256
    assert all(e["payload"]["artifact_sha256"] == model.artifact_sha256
               for e in simulation.journal.read_all() if e["event_type"] == "mapping_allocation_proposed")
    for unit in simulation.units:
        events = unit.adapter.journal.read_all()
        approved = {e["payload"]["command_id"] for e in events
                    if e["event_type"] == "safety_decision" and e["payload"]["outcome"] == "authorised"}
        assert all(e["correlation_id"] in approved for e in events if e["event_type"] == "mapping_vehicle_moved")


def test_benchmark_rejects_test_split_overlap_before_training(tmp_path):
    script = Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"
    benchmark = runpy.run_path(str(script))["benchmark"]
    for seed in (1, 501):
        output = tmp_path / str(seed)
        with pytest.raises(ValueError, match="overlap"):
            benchmark(output, (seed,), ("nominal",))
        assert not output.exists()


def test_benchmark_checksum_binds_the_persisted_report_bytes(tmp_path):
    script = Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"
    benchmark = runpy.run_path(str(script))["benchmark"]
    output = tmp_path / "comparison"
    report = benchmark(output, (9000,), ("nominal",))
    assert all(entry["completion_rate"] == 1 for entry in report["summary"].values())
    assert not report["deployment_approved"]
    assert (output / "report.sha256").read_text(encoding="utf-8").strip() == sha256(
        (output / "report.json").read_bytes()).hexdigest()
