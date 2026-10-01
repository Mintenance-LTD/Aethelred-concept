"""Independent small-case oracle, context fencing, and workload mission evidence."""

import itertools
import json
import math
import runpy
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from aethelred.runtime.geometry import Position
from aethelred.runtime.mapping_allocation import (
    MappingAllocationContext,
    NearestTaskAllocator,
    TaskAssignment,
)
from aethelred.runtime.mapping_tasks import AssignmentError, MappingVehicle, MissionTask, TaskStatus
from aethelred.runtime.operational import OperatingArea
from aethelred.simulation.mapping_mission import SCENARIOS, MappingLayout, MappingMissionSimulation
from aethelred.simulation.workload_allocator import MissionWorkloadAllocator


def task(index, xs):
    return MissionTask(UUID(int=index), UUID(int=100), 1,
                       OperatingArea(Position(0, 0), Position(100, 100)),
                       tuple(Position(x, 0) for x in xs))


def brute_ticks(vehicle, route):
    # Independent test oracle; do not reuse candidate cost/DP helpers.
    current = vehicle.position
    total = 0
    for cell in route:
        for target in cell.samples[cell.completed_samples:]:
            total += max(1, math.ceil(math.hypot(current.x - target.x, current.y - target.y) / vehicle.speed))
            current = target
    return total


@pytest.mark.parametrize("speeds", [(1., 10.), (3., 4.), (2., 3., 7.)])
def test_workload_makespan_matches_independent_exhaustive_oracle(speeds):
    vehicles = tuple(MappingVehicle(str(i), Position(i * 5, 0), .9, speed=s)
                     for i, s in enumerate(speeds))
    cells = (task(1, (2, 6)), task(2, (30, 25)), task(3, (17, 20)))
    candidates = []
    for owners in itertools.product(range(len(vehicles)), repeat=len(cells)):
        assigned = [[t for t, owner in zip(cells, owners) if owner == unit]
                    for unit in range(len(vehicles))]
        for routes in itertools.product(*(itertools.permutations(group) for group in assigned)):
            candidates.append(max(brute_ticks(v, route) for v, route in zip(vehicles, routes)))
    allocator = MissionWorkloadAllocator()
    plan = allocator.plan(vehicles, cells)
    assert plan.projected_ticks == min(candidates)
    assert allocator.plan(tuple(reversed(vehicles)), tuple(reversed(cells))) == plan
    all_planned = [t for _, route in plan.routes for t in route]
    assert set(all_planned) == {t.task_id for t in cells} and len(all_planned) == len(set(all_planned))
    with pytest.raises(FrozenInstanceError):
        plan.projected_ticks = 1


def test_remaining_work_can_leave_slow_unit_idle_instead_of_issuing_bad_batch():
    vehicles = (MappingVehicle("slow", Position(0, 0), .9, speed=1),
                MappingVehicle("fast", Position(0, 0), .9, speed=10))
    cells = (task(1, (1, 5, 10, 15)), task(2, (30,)), task(3, (40,)))
    planner = MissionWorkloadAllocator()
    plan = planner.plan(vehicles, cells)
    assert plan.projected_ticks == 7
    assert dict(plan.routes)["slow"] == ()
    assert planner.propose(vehicles, cells) == (TaskAssignment("fast", UUID(int=1)),)
    assert len(NearestTaskAllocator().propose(vehicles, cells)) == 2


def test_busy_prefix_forecast_cannot_assign_or_steal_from_busy_unit():
    slow = MappingVehicle("slow", Position(0, 0), .9, speed=1)
    fast = MappingVehicle("fast", Position(0, 0), .9, speed=10)
    active = replace(task(4, (1,)), status=TaskStatus.RUNNING, vehicle_id="fast",
                     assignment_version=1, lease_until=datetime.now(UTC) + timedelta(seconds=4))
    pending = (task(1, (1, 5, 10, 15)), task(2, (30,)), task(3, (40,)))
    context = MappingAllocationContext([slow, fast], [active])
    planner = MissionWorkloadAllocator()
    assert planner.propose_with_context((slow,), pending, context) == ()
    plan = planner.plan(context.vehicles, pending, context.active_tasks)
    assert active.task_id not in {t for _, route in plan.routes for t in route}
    with pytest.raises(ValueError, match="cannot own"):
        planner.propose_with_context((fast,), pending, context)
    with pytest.raises(FrozenInstanceError):
        context.vehicles = ()


def test_partial_progress_is_not_replanned_as_unvisited_samples():
    partial = replace(task(1, (1, 5, 10, 15)), completed_samples=2)
    vehicle = MappingVehicle("fast", Position(5, 0), .9, speed=10)
    plan = MissionWorkloadAllocator().plan((vehicle,), (partial,))
    assert plan.projected_ticks == 2 and plan.projected_distance == 10


@pytest.mark.parametrize("mode", ["units", "tasks", "duplicate", "speed", "offline", "exhausted", "samples"])
def test_workload_search_rejects_invalid_or_unbounded_inputs(mode):
    vehicles = (MappingVehicle("unit", Position(0, 0), .9),)
    cells = (task(1, (1,)),)
    if mode == "units":
        vehicles = tuple(replace(vehicles[0], vehicle_id=str(i)) for i in range(4))
    elif mode == "tasks":
        cells = tuple(task(i, (1,)) for i in range(1, 14))
    elif mode == "duplicate":
        cells = cells * 2
    elif mode == "speed":
        vehicles = (replace(vehicles[0], speed=float("nan")),)
    elif mode == "offline":
        vehicles = (replace(vehicles[0], available=False),)
    elif mode == "exhausted":
        cells = (replace(cells[0], completed_samples=1),)
    elif mode == "samples":
        cells = (task(1, (1, 2, 3, 4, 5)),)
    with pytest.raises(ValueError):
        MissionWorkloadAllocator().plan(vehicles, cells)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_workload_missions_keep_fault_recovery_and_authorised_command_path(tmp_path, scenario):
    planner = MissionWorkloadAllocator()
    simulation = MappingMissionSimulation(tmp_path / scenario, scenario,
                                          layout=MappingLayout.from_seed(10500), allocator=planner)
    initial_plan = planner.plan(tuple(MappingVehicle(u.adapter.vehicle_id, u.adapter.position,
                                                     u.adapter.battery, speed=u.adapter.speed)
                                      for u in simulation.units), simulation.coordinator.tasks)
    result = simulation.run()
    assert result.completed and result.visited_samples == result.total_samples
    assert result.duplicate_visits == 0
    if scenario == "nominal":
        assert result.ticks == initial_plan.projected_ticks
    if scenario in {"unit_loss", "comms_loss"}:
        assert result.expired_assignments >= 1
    if scenario == "comms_loss":
        assert result.stale_assignment_rejections == 1
    events = simulation.journal.read_all()
    proposals = [e["payload"] for e in events if e["event_type"] == "mapping_allocation_proposed"]
    assert proposals and all(p["artifact_sha256"] == planner.artifact_sha256 for p in proposals)
    assert all("observed_context" in p for p in proposals)
    benchmark = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_mapping_allocators.py"))
    assert benchmark["boundary_violations"](simulation) == 0


def test_contextual_allocator_still_cannot_grant_a_busy_or_unknown_assignment(tmp_path):
    class UntrustedContextPlanner:
        allocator_id = "untrusted-context"
        artifact_sha256 = None

        def propose(self, vehicles, tasks):
            pytest.fail("Context extension should be selected")

        def propose_with_context(self, idle, pending, context):
            assert "mapper-1" not in {v.vehicle_id for v in idle}
            assert "mapper-1" in {v.vehicle_id for v in context.vehicles}
            return (TaskAssignment("mapper-1", pending[0].task_id),)

    simulation = MappingMissionSimulation(tmp_path / "context", "nominal")
    now = simulation.started_at
    vehicles = tuple(MappingVehicle(u.adapter.vehicle_id, u.adapter.position, .9) for u in simulation.units)
    simulation.coordinator.allocate(vehicles[:1], now)
    before = simulation.coordinator.tasks
    simulation.coordinator.allocator = UntrustedContextPlanner()
    with pytest.raises(AssignmentError):
        simulation.coordinator.allocate(vehicles, now)
    assert simulation.coordinator.tasks == before


def test_preserved_workload_evidence_reproduces_the_four_way_comparison():
    from statistics import mean

    project = Path(__file__).parents[1]
    snapshot = json.loads((project / "experiments/mapping_allocator/WORKLOAD_RESULTS.json")
                          .read_text(encoding="utf-8"))
    columns = snapshot["measurement_columns"]
    assert len(columns) == len(set(columns))
    rows = []
    for values in snapshot["measurements"]:
        assert len(values) == len(columns)
        fields = dict(zip(columns, values))
        result = {k.removeprefix("result."): v for k, v in fields.items() if k.startswith("result.")}
        assert result["scenario"] == fields["scenario"]
        assert result["completed"] and result["visited_samples"] == result["total_samples"]
        assert result["duplicate_visits"] == fields["boundary_violations"] == 0
        assert fields["planning_wall_time.calls"] >= 1
        assert 0 <= fields["planning_wall_time.max_seconds"] <= fields["planning_wall_time.total_seconds"]
        rows.append({"seed": fields["seed"], "scenario": fields["scenario"],
                     "allocator": fields["allocator"], "result": result})
    assert len(rows) == 320 and len(snapshot["layouts"]) == 16
    assert set(snapshot["held_out_seeds"]) == set(range(11000, 11016))
    assert snapshot["candidate"] == "mission-workload"
    assert snapshot["candidate_sha256"] == snapshot["source_file_sha256"]["workload_allocator.py"]
    assert snapshot["frozen_comparator_model_sha256"] == "79f7e19dc18cc7db97c9516ac8a922ab46c65f929525311777efa7f556d29533"
    for used in (snapshot["training"]["training_seeds"], snapshot["training"]["validation_seeds"],
                 snapshot["evaluation_plan"]["excluded_seeds"]):
        assert set(snapshot["held_out_seeds"]).isdisjoint(used)
    for name, summary in snapshot["summary"].items():
        results = [r["result"] for r in rows if r["allocator"] == name]
        assert len(results) == summary["mission_count"] == 80
        assert mean(r["ticks"] for r in results) == pytest.approx(summary["mean_ticks"])
        assert mean(r["distance_travelled"] for r in results) == pytest.approx(summary["mean_distance"])
    paired_analysis = runpy.run_path(str(project / "scripts/benchmark_mapping_allocators.py"))["paired_analysis"]
    reproduced = paired_analysis(rows, tuple(snapshot["held_out_seeds"]), tuple(snapshot["scenarios"]),
                                 candidate="mission-workload",
                                 baselines=("nearest-task", "analytic-balanced", "learned-balanced"))
    for baseline, metrics in reproduced["comparisons"].items():
        for metric, stats in metrics.items():
            expected = snapshot["paired_analysis"]["comparisons"][baseline][metric]
            assert stats["mean_paired_difference"] == pytest.approx(expected["mean_paired_difference"])
            assert stats["bootstrap_95_percent_interval"] == pytest.approx(expected["bootstrap_95_percent_interval"])
    assert snapshot["candidate_clears_local_comparison"] and not snapshot["deployment_approved"]
