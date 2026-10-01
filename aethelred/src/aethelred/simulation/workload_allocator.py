"""Bounded, simulation-only remaining-work planner for civilian sample mapping.

Held-Karp routes and exhaustive fleet partitions minimize projected makespan
under current observed speeds and locked task prefixes. Distance breaks ties
between the minimum-time routes retained for each unit/subset. This is not a
global Pareto optimizer for travel, a fault oracle, or a vehicle controller.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from aethelred.runtime.geometry import Position
from aethelred.runtime.mapping_allocation import MappingAllocationContext, TaskAssignment
from aethelred.runtime.mapping_tasks import MappingVehicle, MissionTask, TaskStatus


@dataclass(frozen=True)
class WorkloadPlan:
    routes: tuple[tuple[str, tuple[UUID, ...]], ...]
    projected_ticks: int
    projected_distance: float


# Minimum ticks, then distance, then canonical task indexes.
Route = tuple[int, float, tuple[int, ...]]


def _segment(start: Position, end: Position, speed: float) -> tuple[int, float]:
    distance = start.distance_to(end)
    return max(1, math.ceil(distance / speed)), distance


def _task_cost(start: Position, task: MissionTask, speed: float) -> tuple[int, float]:
    segments = [_segment(a, b, speed) for a, b in
                itertools.pairwise((start, *task.samples[task.completed_samples:]))]
    return sum(ticks for ticks, _ in segments), sum(distance for _, distance in segments)


def _subset_routes(vehicle: MappingVehicle, tasks: tuple[MissionTask, ...],
                   prefix: MissionTask | None) -> list[Route]:
    """Exact minimum-duration route for every pending subset, with locked prefix."""
    count = len(tasks)
    start = prefix.samples[-1] if prefix is not None else vehicle.position
    initial = _task_cost(vehicle.position, prefix, vehicle.speed) if prefix is not None else (0, 0.)
    arrivals = [_task_cost(start, task, vehicle.speed) for task in tasks]
    transitions = [[_task_cost(left.samples[-1], right, vehicle.speed)
                    for right in tasks] for left in tasks]
    states: dict[tuple[int, int], Route] = {
        (1 << index, index): (ticks + initial[0], distance + initial[1], (index,))
        for index, (ticks, distance) in enumerate(arrivals)}
    routes: list[Route] = [(initial[0], initial[1], ())] * (1 << count)
    for mask in range(1, 1 << count):
        best: Route | None = None
        for last in range(count):
            if not mask & (1 << last):
                continue
            previous = mask ^ (1 << last)
            if previous:
                choices = []
                for before in range(count):
                    if not previous & (1 << before):
                        continue
                    ticks, distance, route = states[previous, before]
                    travel_ticks, travel_distance = transitions[before][last]
                    choices.append((ticks + travel_ticks, distance + travel_distance, (*route, last)))
                states[mask, last] = min(choices)
            current = states[mask, last]
            if best is None or current < best:
                best = current
        assert best is not None
        routes[mask] = best
    return routes


class MissionWorkloadAllocator:
    """Propose only the first task for each currently idle unit in the full plan.

    Busy units keep their leases and must finish their current routes. Future
    routes are forecasts, never reservations. Replanning uses observations only;
    missing units and unknown future faults are not assumed to recover.
    """

    allocator_id = "mission-workload/v1"

    def __init__(self) -> None:
        # Deterministic planner artifact, not trained weights or a promoted release.
        self.artifact_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()

    def plan(self, vehicles: tuple[MappingVehicle, ...], tasks: tuple[MissionTask, ...],
             active_tasks: tuple[MissionTask, ...] = ()) -> WorkloadPlan:
        if len(vehicles) > 3 or len(tasks) + len(active_tasks) > 12:
            raise ValueError("Workload planning supports at most 3 units and 12 unfinished cells")
        if len({v.vehicle_id for v in vehicles}) != len(vehicles):
            raise ValueError("Workload vehicle IDs must be unique")
        if (any(not v.available or not math.isfinite(v.speed) or v.speed <= 0
                or not math.isfinite(v.position.x) or not math.isfinite(v.position.y)
                for v in vehicles)):
            raise ValueError("Workload planning requires available units with finite positions and positive speeds")
        all_tasks = (*tasks, *active_tasks)
        if any(len(t.samples) > 4 for t in all_tasks):
            raise ValueError("Workload planning supports at most 4 samples per cell")
        if len({t.task_id for t in all_tasks}) != len(all_tasks):
            raise ValueError("Workload task IDs must be unique")
        if any(t.status is not TaskStatus.PENDING or t.completed_samples >= len(t.samples) for t in tasks):
            raise ValueError("Workload planning requires pending, non-exhausted routes")
        if any(t.status not in {TaskStatus.ASSIGNED, TaskStatus.RUNNING} for t in active_tasks):
            raise ValueError("Workload prefixes must be active assignments")
        if len({t.vehicle_id for t in active_tasks}) != len(active_tasks):
            raise ValueError("Workload planning requires at most one locked route per unit")
        if tasks and not vehicles:
            raise ValueError("Pending workload requires an observed unit")
        units = tuple(sorted(vehicles, key=lambda v: v.vehicle_id))
        pending = tuple(sorted(tasks, key=lambda t: str(t.task_id)))
        owned = {t.vehicle_id: t for t in active_tasks}
        costs = [_subset_routes(v, pending, owned.get(v.vehicle_id)) for v in units]
        full = (1 << len(pending)) - 1
        best_key: tuple[int, float, tuple[tuple[int, ...], ...]] | None = None
        best_routes: tuple[tuple[int, ...], ...] = ()

        def consider(masks: tuple[int, ...]) -> None:
            nonlocal best_key, best_routes
            selected = tuple(cost[mask] for cost, mask in zip(costs, masks, strict=True))
            routes = tuple(route for _, _, route in selected)
            key = (max((ticks for ticks, _, _ in selected), default=0),
                   sum(distance for _, distance, _ in selected), routes)
            if best_key is None or key < best_key:
                best_key, best_routes = key, routes

        if len(units) <= 1:
            consider((full,) if units else ())
        elif len(units) == 2:
            for first in range(full + 1):
                consider((first, full ^ first))
        else:
            for first in range(full + 1):
                remaining = full ^ first
                second = remaining
                while True:
                    consider((first, second, remaining ^ second))
                    if not second:
                        break
                    second = (second - 1) & remaining
        assert best_key is not None
        return WorkloadPlan(tuple((v.vehicle_id, tuple(pending[i].task_id for i in route))
                                  for v, route in zip(units, best_routes, strict=True)),
                            best_key[0], best_key[1])

    def propose(self, vehicles: tuple[MappingVehicle, ...],
                tasks: tuple[MissionTask, ...]) -> tuple[TaskAssignment, ...]:
        return self.propose_with_context(vehicles, tasks, MappingAllocationContext(vehicles, ()))

    def propose_with_context(self, idle_vehicles: tuple[MappingVehicle, ...],
                             pending_tasks: tuple[MissionTask, ...],
                             context: MappingAllocationContext) -> tuple[TaskAssignment, ...]:
        idle = {v.vehicle_id for v in idle_vehicles}
        fleet = {v.vehicle_id for v in context.vehicles}
        owned = {t.vehicle_id for t in context.active_tasks}
        if not idle <= fleet or idle & owned:
            raise ValueError("Idle units must belong to the observed fleet and cannot own active tasks")
        plan = self.plan(context.vehicles, pending_tasks, context.active_tasks)
        return tuple(TaskAssignment(vehicle, route[0]) for vehicle, route in plan.routes
                     if vehicle in idle and route)
