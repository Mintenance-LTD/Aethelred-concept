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


def test_frozen_benchmark_never_retrains_and_preserves_exact_artifact(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"
    benchmark = runpy.run_path(str(script))["benchmark"]
    model_path = tmp_path / "frozen.json"
    train_duration_model(model_path, (1, 2), (501,))
    original = model_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Frozen evaluation must never call training")

    def fake_case(case):
        _, seed, scenario, name, _ = case
        return {"seed": seed, "scenario": scenario, "allocator": name,
                "result": {"completed": True, "ticks": 10, "distance_travelled": 20.,
                           "duplicate_visits": 0, "visited_samples": 4, "total_samples": 4},
                "boundary_violations": 0}

    monkeypatch.setitem(benchmark.__globals__, "train_duration_model", forbidden)
    monkeypatch.setitem(benchmark.__globals__, "run_case", fake_case)
    output = tmp_path / "evaluation"
    report = benchmark(output, (10000, 10001), ("nominal",), model_path=model_path,
                       expected_sha256=sha256(original).hexdigest(), excluded_seeds=(9100,))
    assert model_path.read_bytes() == (output / "duration-model.json").read_bytes() == original
    assert report["evaluation_plan"]["frozen_model"]
    assert report["paired_analysis"]["independent_layout_count"] == 2
    assert not report["candidate_clears_local_comparison"]


@pytest.mark.parametrize("mode", ["missing_hash", "wrong_hash", "training", "validation", "previous"])
def test_frozen_benchmark_rejects_bad_fingerprint_or_overlap_before_output(tmp_path, mode):
    script = Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"
    benchmark = runpy.run_path(str(script))["benchmark"]
    path = tmp_path / "model.json"
    train_duration_model(path, (1,), (501,))
    digest = sha256(path.read_bytes()).hexdigest()
    seed = {"training": 1, "validation": 501, "previous": 9100}.get(mode, 10000)
    fingerprint = None if mode == "missing_hash" else "0" * 64 if mode == "wrong_hash" else digest
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="SHA256|overlap"):
        benchmark(output, (seed,), ("nominal",), model_path=path,
                  expected_sha256=fingerprint, excluded_seeds=(9100,))
    assert not output.exists()


def test_paired_statistics_resample_layouts_not_scenario_variants():
    script = Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"
    paired_analysis = runpy.run_path(str(script))["paired_analysis"]
    rows = [{"seed": seed, "scenario": scenario, "allocator": name,
             "result": {"ticks": 10 + (delta if name == "learned-balanced" else 0),
                        "distance_travelled": 100.}}
            for seed, delta in ((10000, -2), (10001, 2))
            for scenario in ("nominal", "unit_loss")
            for name in ("nearest-task", "analytic-balanced", "learned-balanced")]
    report = paired_analysis(rows, (10000, 10001), ("nominal", "unit_loss"))
    assert report["independent_layout_count"] == 2
    stats = report["comparisons"]["nearest-task"]["ticks"]
    assert stats["mean_paired_difference"] == 0
    assert stats["bootstrap_95_percent_interval"] == [-2., 2.]
    assert stats["layouts_faster_or_shorter"] == stats["layouts_slower_or_longer"] == 1
    assert report == paired_analysis(rows, (10000, 10001), ("nominal", "unit_loss"))
    with pytest.raises(ValueError, match="exactly one"):
        paired_analysis(rows + rows[:1], (10000, 10001), ("nominal", "unit_loss"))
    with pytest.raises(ValueError, match="exactly one"):
        paired_analysis(rows[:-1], (10000, 10001), ("nominal", "unit_loss"))


def test_preserved_independent_measurements_reproduce_reported_comparison():
    from statistics import mean

    project = Path(__file__).parents[1]
    snapshot = json.loads((project / "experiments/mapping_allocator/INDEPENDENT_RESULTS.json")
                          .read_text(encoding="utf-8"))
    columns = snapshot["measurement_columns"]
    assert len(set(columns)) == len(columns)
    rows = []
    for values in snapshot["measurements"]:
        assert len(values) == len(columns)
        fields = dict(zip(columns, values))
        rows.append({"seed": fields["seed"], "scenario": fields["scenario"],
                     "allocator": fields["allocator"],
                     "result": {k.removeprefix("result."): v for k, v in fields.items()
                                if k.startswith("result.")},
                     "boundary_violations": fields["boundary_violations"]})
    assert len(rows) == 240
    assert snapshot["evaluation_plan"]["frozen_model"]
    assert set(snapshot["held_out_seeds"]) == set(range(10000, 10016))
    for split in ("training_seeds", "validation_seeds"):
        assert set(snapshot["held_out_seeds"]).isdisjoint(snapshot["training"][split])
    assert set(snapshot["held_out_seeds"]).isdisjoint(snapshot["evaluation_plan"]["excluded_seeds"])
    for name, summary in snapshot["summary"].items():
        results = [r["result"] for r in rows if r["allocator"] == name]
        assert len(results) == summary["mission_count"] == 80
        assert all(r["completed"] and r["visited_samples"] == r["total_samples"] for r in results)
        assert mean(r["ticks"] for r in results) == pytest.approx(summary["mean_ticks"])
        assert mean(r["distance_travelled"] for r in results) == pytest.approx(summary["mean_distance"])
        assert sum(r["duplicate_visits"] for r in results) == summary["duplicate_visits"] == 0
    assert all(r["boundary_violations"] == 0 for r in rows)
    paired_analysis = runpy.run_path(str(project / "scripts/benchmark_mapping_allocators.py"))["paired_analysis"]
    reproduced = paired_analysis(rows, tuple(snapshot["held_out_seeds"]), tuple(snapshot["scenarios"]))
    for baseline, metrics in reproduced["comparisons"].items():
        for metric, stats in metrics.items():
            expected = snapshot["paired_analysis"]["comparisons"][baseline][metric]
            assert stats["mean_paired_difference"] == pytest.approx(expected["mean_paired_difference"])
            assert stats["bootstrap_95_percent_interval"] == pytest.approx(expected["bootstrap_95_percent_interval"])
    assert not snapshot["candidate_clears_local_comparison"] and not snapshot["deployment_approved"]
