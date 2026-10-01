"""Proposal-only allocation contracts; task ownership stays with the coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
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


@dataclass(frozen=True)
class MappingAllocationContext:
    """Read-only observed fleet and locked work, not future ownership grants."""

    vehicles: tuple[MappingVehicle, ...]
    active_tasks: tuple[MissionTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vehicles", tuple(self.vehicles))
        object.__setattr__(self, "active_tasks", tuple(self.active_tasks))


@runtime_checkable
class ContextualMappingAllocator(Protocol):
    """Optional extension; existing two-argument allocators remain compatible."""

    def propose_with_context(self, idle_vehicles: tuple[MappingVehicle, ...],
                             pending_tasks: tuple[MissionTask, ...],
                             context: MappingAllocationContext) -> tuple[TaskAssignment, ...]: ...


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
