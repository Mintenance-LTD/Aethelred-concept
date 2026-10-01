"""Proposal-only allocation contracts; task ownership stays with the coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from aethelred.runtime.mapping_tasks import MappingVehicle, MissionTask


@dataclass(frozen=True)
class TaskAssignment:
    vehicle_id: str
    task_id: UUID


class MappingAllocator(Protocol):
    allocator_id: str
    artifact_sha256: str | None

    def propose(self, vehicles: tuple[MappingVehicle, ...],
                tasks: tuple[MissionTask, ...]) -> tuple[TaskAssignment, ...]: ...


class NearestTaskAllocator:
    """The original deterministic baseline: nearest pending task per idle unit."""

    allocator_id = "nearest-task/v1"
    artifact_sha256 = None

    def propose(self, vehicles: tuple[MappingVehicle, ...],
                tasks: tuple[MissionTask, ...]) -> tuple[TaskAssignment, ...]:
        pending = list(tasks)
        assignments = []
        for vehicle in sorted(vehicles, key=lambda v: v.vehicle_id):
            if not pending:
                break
            task = min(pending, key=lambda t: (vehicle.position.distance_to(t.target), str(t.task_id)))
            assignments.append(TaskAssignment(vehicle.vehicle_id, task.task_id))
            pending.remove(task)
        return tuple(assignments)
