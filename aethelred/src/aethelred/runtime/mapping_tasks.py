"""Single-coordinator, durable task ownership for bounded mapping missions."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from uuid import UUID, uuid5

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.geometry import Position
from aethelred.runtime.mapping_allocation import (
    MappingAllocator,
    NearestTaskAllocator,
    TaskAssignment,
)
from aethelred.runtime.operational import Mission, MissionCapability, OperatingArea


class TaskStatus(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"


class AssignmentError(PermissionError):
    """A message does not own the current, unexpired assignment."""


@dataclass(frozen=True)
class MissionTask:
    task_id: UUID
    mission_id: UUID
    mission_revision: int
    area: OperatingArea
    samples: tuple[Position, ...]
    completed_samples: int = 0
    status: TaskStatus = TaskStatus.PENDING
    vehicle_id: str | None = None
    assignment_version: int = 0
    lease_until: datetime | None = None

    def __post_init__(self) -> None:
        if self.mission_revision < 1 or self.assignment_version < 0 or not self.samples:
            raise ValueError("Invalid task revision, version, or empty sample route")
        if not 0 <= self.completed_samples <= len(self.samples):
            raise ValueError("Invalid task sample progress")
        if any(not self.area.contains(p) for p in self.samples):
            raise ValueError("Task samples must be inside their cell")
        if self.status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}:
            if not self.vehicle_id or self.assignment_version < 1 or self.lease_until is None:
                raise ValueError("Owned tasks require a vehicle, version, and lease")
            if self.lease_until.tzinfo is None or self.completed_samples == len(self.samples):
                raise ValueError("Invalid lease or exhausted active task")
        elif self.lease_until is not None:
            raise ValueError("Unowned or completed tasks cannot retain a lease")
        if self.status is TaskStatus.COMPLETED and self.completed_samples != len(self.samples):
            raise ValueError("Completed task must include all samples")
        if self.status is TaskStatus.PENDING and self.vehicle_id is not None:
            raise ValueError("Pending task cannot have an owner")

    @property
    def target(self) -> Position:
        return self.samples[self.completed_samples]


@dataclass(frozen=True)
class MappingVehicle:
    vehicle_id: str
    position: Position
    battery_reserve: float
    available: bool = True
    speed: float = 8.0


class MappingCoordinator:
    """Allocate sample routes and fence stale messages using expiring leases.

    One coordinator owns this journal at a time. Recovery preserves versions and
    progress; it never automatically extends a previously issued lease.
    """

    def __init__(self, mission: Mission, journal: JsonlAuditJournal,
                 lease_duration: timedelta = timedelta(seconds=4),
                 allocator: MappingAllocator | None = None) -> None:
        if MissionCapability.MAP not in mission.allowed_capabilities:
            raise ValueError("Mapping requires the mission MAP capability")
        if lease_duration <= timedelta():
            raise ValueError("Task leases must be positive")
        self.mission = mission
        self.journal = journal
        self.lease_duration = lease_duration
        self.allocator = allocator or NearestTaskAllocator()
        self._tasks: dict[UUID, MissionTask] = {}
        for event in journal.read_all():
            if event["event_type"] == "mapping_grid_created":
                payload = event["payload"]
                if payload["mission_id"] != str(mission.mission_id):
                    continue
                if self._tasks:
                    raise ValueError("Duplicate mapping grid in task journal")
                tasks = tuple(self._decode(raw) for raw in payload["tasks"])
                if not tasks or len({t.task_id for t in tasks}) != len(tasks):
                    raise ValueError("Invalid mapping grid")
                for task in tasks:
                    self._validate_mission(task)
                    self._tasks[task.task_id] = task
                continue
            if not str(event["event_type"]).startswith("mapping_task_"):
                continue
            payload = event["payload"]
            if payload.get("mission_id") != str(mission.mission_id):
                continue
            task = self._decode(payload)
            self._validate_mission(task)
            if task.task_id not in self._tasks:
                raise ValueError("Task transition has no durable grid definition")
            self._tasks[task.task_id] = task

    @property
    def tasks(self) -> tuple[MissionTask, ...]:
        return tuple(self._tasks.values())

    @property
    def complete(self) -> bool:
        return bool(self._tasks) and all(t.status is TaskStatus.COMPLETED for t in self.tasks)

    def create_grid(self, columns: int = 3, rows: int = 2) -> None:
        """Partition the area into cells, each with four explicit sample locations."""
        if self._tasks:
            raise ValueError("Mission tasks already exist")
        if type(columns) is not int or type(rows) is not int or columns < 1 or rows < 1:
            raise ValueError("Grid dimensions must be positive integers")
        area = self.mission.operating_area
        width = (area.maximum.x - area.minimum.x) / columns
        height = (area.maximum.y - area.minimum.y) / rows
        if width <= 0 or height <= 0:
            raise ValueError("Mapping area must have positive dimensions")
        tasks = []
        for row in range(rows):
            for column in range(columns):
                x, y = area.minimum.x + column * width, area.minimum.y + row * height
                right = area.maximum.x if column == columns - 1 else x + width
                top = area.maximum.y if row == rows - 1 else y + height
                cell = OperatingArea(Position(x, y), Position(right, top))
                samples = tuple(Position(x + dx * width, y + dy * height)
                                for dx, dy in ((.25, .25), (.75, .25), (.75, .75), (.25, .75)))
                task = MissionTask(uuid5(self.mission.mission_id, f"{self.mission.revision}:{column}:{row}"),
                                   self.mission.mission_id, self.mission.revision, cell, samples)
                tasks.append(task)
        # One durable record prevents a crash from creating a partial mission grid.
        self.journal.record("mapping_grid_created", str(self.mission.mission_id), {
            "mission_id": str(self.mission.mission_id), "columns": columns, "rows": rows,
            "tasks": [self._encode(task) for task in tasks],
        })
        self._tasks.update((task.task_id, task) for task in tasks)

    def allocate(self, vehicles: tuple[MappingVehicle, ...], now: datetime) -> None:
        self._check_time(now)
        self.expire(now)
        occupied = {t.vehicle_id for t in self.tasks if t.status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}}
        eligible = [v for v in vehicles if v.available and v.vehicle_id not in occupied
                    and v.vehicle_id in self.mission.assigned_vehicle_ids
                    and isfinite(v.battery_reserve) and .25 <= v.battery_reserve <= 1
                    and isfinite(v.speed) and v.speed > 0
                    and self.mission.operating_area.contains(v.position)]
        if len({v.vehicle_id for v in vehicles}) != len(vehicles):
            raise ValueError("Vehicle observations must be unique")
        pending = {t.task_id: t for t in self.tasks if t.status is TaskStatus.PENDING}
        if not eligible or not pending:
            return
        proposals = tuple(self.allocator.propose(tuple(sorted(eligible, key=lambda v: v.vehicle_id)),
                                                 tuple(pending.values())))
        known_vehicles = {v.vehicle_id for v in eligible}
        seen_vehicles: set[str] = set()
        seen_tasks: set[UUID] = set()
        for proposal in proposals:
            if (not isinstance(proposal, TaskAssignment) or proposal.vehicle_id not in known_vehicles
                    or proposal.task_id not in pending or proposal.vehicle_id in seen_vehicles
                    or proposal.task_id in seen_tasks):
                self.journal.record("mapping_allocation_rejected", str(self.mission.mission_id),
                                    {"allocator_id": self.allocator.allocator_id, "reason": "invalid_assignment"})
                raise AssignmentError("Allocator proposed an unavailable, duplicate, or unknown assignment")
            seen_vehicles.add(proposal.vehicle_id)
            seen_tasks.add(proposal.task_id)
        self.journal.record("mapping_allocation_proposed", str(self.mission.mission_id), {
            "allocator_id": self.allocator.allocator_id,
            "artifact_sha256": self.allocator.artifact_sha256,
            "assignments": [{"vehicle_id": p.vehicle_id, "task_id": str(p.task_id)} for p in proposals],
        })
        for proposal in proposals:
            task = pending[proposal.task_id]
            self._persist("assigned", replace(task, status=TaskStatus.ASSIGNED,
                          vehicle_id=proposal.vehicle_id, assignment_version=task.assignment_version + 1,
                          lease_until=min(now + self.lease_duration, self.mission.valid_until)))

    def expire(self, now: datetime) -> None:
        for task in self.tasks:
            if task.lease_until is not None and now >= task.lease_until:
                self._persist("expired", replace(task, status=TaskStatus.PENDING,
                              vehicle_id=None, lease_until=None))

    def owned(self, task_id: UUID, vehicle_id: str, version: int, now: datetime) -> MissionTask:
        self._check_time(now)
        task = self._tasks[task_id]
        if (task.vehicle_id != vehicle_id or task.assignment_version != version
                or task.status not in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}
                or task.lease_until is None or now >= task.lease_until):
            raise AssignmentError("Task assignment is stale, expired, or owned by another vehicle")
        return task

    def acknowledge(self, task_id: UUID, vehicle_id: str, version: int, now: datetime) -> None:
        task = self.owned(task_id, vehicle_id, version, now)
        if task.status is TaskStatus.ASSIGNED:
            self._persist("acknowledged", replace(task, status=TaskStatus.RUNNING))

    def heartbeat(self, task_id: UUID, vehicle_id: str, version: int, now: datetime) -> None:
        task = self.owned(task_id, vehicle_id, version, now)
        self._persist("heartbeat", replace(task, lease_until=min(now + self.lease_duration,
                                                               self.mission.valid_until)))

    def cancel(self, task_id: UUID, vehicle_id: str, version: int, now: datetime) -> None:
        """Release an acknowledged assignment without discarding recorded progress."""
        task = self.owned(task_id, vehicle_id, version, now)
        self._persist("cancelled", replace(task, status=TaskStatus.PENDING,
                      vehicle_id=None, lease_until=None))

    def record_sample(self, task_id: UUID, vehicle_id: str, version: int,
                      sample_index: int, now: datetime) -> None:
        task = self.owned(task_id, vehicle_id, version, now)
        if type(sample_index) is not int or sample_index < 0:
            raise AssignmentError("Sample index must be a non-negative integer")
        if task.status is not TaskStatus.RUNNING:
            raise AssignmentError("Task must be acknowledged before reporting progress")
        if sample_index < task.completed_samples:
            return  # Idempotent duplicate progress from the current owner.
        if sample_index != task.completed_samples:
            raise AssignmentError("Sample progress must be contiguous")
        completed = task.completed_samples + 1
        finished = completed == len(task.samples)
        self._persist("completed" if finished else "progress", replace(
            task, completed_samples=completed,
            status=TaskStatus.COMPLETED if finished else TaskStatus.RUNNING,
            lease_until=None if finished else task.lease_until))

    def _check_time(self, now: datetime) -> None:
        if now.tzinfo is None or not self.mission.valid_from <= now < self.mission.valid_until:
            raise AssignmentError("Mission is not valid at the supplied time")

    def _persist(self, event: str, task: MissionTask) -> None:
        self.journal.record(f"mapping_task_{event}", str(task.task_id), self._encode(task))
        self._tasks[task.task_id] = task

    def _validate_mission(self, task: MissionTask) -> None:
        if (task.mission_id, task.mission_revision) != (self.mission.mission_id, self.mission.revision):
            raise ValueError("Task journal belongs to another mission revision")
        if not all(self.mission.operating_area.contains(p) for p in (task.area.minimum, task.area.maximum)):
            raise ValueError("Task cell is outside the mission area")

    @staticmethod
    def _encode(task: MissionTask) -> dict:
        return {
            "task_id": str(task.task_id), "mission_id": str(task.mission_id),
            "mission_revision": task.mission_revision,
            "minimum": {"x": task.area.minimum.x, "y": task.area.minimum.y},
            "maximum": {"x": task.area.maximum.x, "y": task.area.maximum.y},
            "samples": [{"x": p.x, "y": p.y} for p in task.samples],
            "completed_samples": task.completed_samples, "status": task.status.value,
            "vehicle_id": task.vehicle_id, "assignment_version": task.assignment_version,
            "lease_until": task.lease_until.isoformat() if task.lease_until else None,
        }

    @staticmethod
    def _decode(raw: dict) -> MissionTask:
        return MissionTask(UUID(raw["task_id"]), UUID(raw["mission_id"]), raw["mission_revision"],
                           OperatingArea(Position(**raw["minimum"]), Position(**raw["maximum"])),
                           tuple(Position(**p) for p in raw["samples"]), raw["completed_samples"],
                           TaskStatus(raw["status"]), raw["vehicle_id"], raw["assignment_version"],
                           datetime.fromisoformat(raw["lease_until"]) if raw["lease_until"] else None)
