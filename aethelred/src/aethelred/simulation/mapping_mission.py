"""Civilian mapping demonstration using the authenticated operational boundary.

Coverage means visitation of prescribed sample locations in this abstract 2-D
simulator. It does not establish camera coverage, flight dynamics, or SIL evidence.
"""

import json
import random
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from html import escape
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from aethelred.runtime.audit import JsonlAuditJournal
from aethelred.runtime.configuration import RuntimeConfigurationRegistry
from aethelred.runtime.geometry import Position
from aethelred.runtime.health import RuntimeHealthReport, RuntimeHealthSupervisor
from aethelred.runtime.integrity import IntentAuthenticator
from aethelred.runtime.lifecycle import RuntimeLifecycleSupervisor
from aethelred.runtime.mapping_allocation import MappingAllocator
from aethelred.runtime.mapping_tasks import (
    AssignmentError,
    MappingCoordinator,
    MappingVehicle,
    TaskStatus,
)
from aethelred.runtime.missions import MissionRegistry
from aethelred.runtime.operational import (
    AuthenticatedOperationalControlLoop,
    AuthorisedCommand,
    CommandReceipt,
    IntentProposal,
    Mission,
    MissionCapability,
    ObservationProvenance,
    OperatingArea,
    OperationalControlLoop,
    OperationalSafetySupervisor,
    RuntimeIdentity,
    WorldState,
)

SCENARIOS = ("nominal", "unit_loss", "comms_loss", "stale_sensor", "coordinator_restart")


@dataclass(frozen=True)
class MappingLayout:
    """Reproducible mission geometry and initial vehicle conditions."""

    seed: int | None = None
    width: float = 60.0
    height: float = 40.0
    columns: int = 3
    rows: int = 2
    positions: tuple[Position, ...] = (Position(2, 2), Position(30, 2), Position(58, 2))
    speeds: tuple[float, ...] = (8.0, 8.0, 8.0)

    def __post_init__(self) -> None:
        if not 20 <= self.width <= 100 or not 20 <= self.height <= 80:
            raise ValueError("Layout dimensions are outside the supported simulation range")
        if not 1 <= self.columns <= 4 or not 1 <= self.rows <= 3:
            raise ValueError("Layout grid dimensions are outside the supported range")
        if len(self.positions) != 3 or len(self.speeds) != 3:
            raise ValueError("Mapping layout requires exactly three units")
        area = OperatingArea(Position(0, 0), Position(self.width, self.height))
        if any(not area.contains(p) for p in self.positions) or any(not 0 < s <= 20 for s in self.speeds):
            raise ValueError("Invalid initial vehicle positions or speeds")

    @classmethod
    def from_seed(cls, seed: int) -> "MappingLayout":
        rng = random.Random(seed)
        width, height = rng.uniform(40, 90), rng.uniform(25, 60)
        return cls(seed, width, height, rng.choice((2, 3, 4)), rng.choice((2, 3)),
                   tuple(Position(rng.uniform(0, width), rng.uniform(0, height)) for _ in range(3)),
                   tuple(rng.uniform(4, 12) for _ in range(3)))


@dataclass
class MappingAdapter:
    """A bounded movement model; the runtime is its only command caller."""

    vehicle_id: str
    position: Position
    journal: JsonlAuditJournal
    now: datetime
    speed: float = 8.0
    battery: float = .95

    def execute(self, command: AuthorisedCommand) -> CommandReceipt:
        if (command.vehicle_id != self.vehicle_id or command.capability is not MissionCapability.MAP
                or command.target_position is None or command.expires_at <= self.now):
            return CommandReceipt(command.command_id, False, self.now, command.sequence, "Invalid mapping command")
        target = Position.copy_of(command.target_position)
        distance = self.position.distance_to(target)
        fraction = min(1.0, self.speed / distance) if distance else 1.0
        position = Position(self.position.x + fraction * (target.x - self.position.x),
                            self.position.y + fraction * (target.y - self.position.y))
        self.journal.record("mapping_vehicle_moved", str(command.command_id), {
            "simulated_at": self.now, "vehicle_id": self.vehicle_id,
            "x": position.x, "y": position.y, "sequence": command.sequence,
        })
        self.position = position
        self.battery -= .001
        return CommandReceipt(command.command_id, True, self.now, command.sequence, "Simulated move accepted")


@dataclass
class MappingUnit:
    adapter: MappingAdapter
    loop: AuthenticatedOperationalControlLoop
    authenticator: IntentAuthenticator
    health: RuntimeHealthSupervisor
    online: bool = True


@dataclass(frozen=True)
class MappingResult:
    scenario: str
    completed: bool
    ticks: int
    completed_tasks: int
    total_tasks: int
    visited_samples: int
    total_samples: int
    duplicate_visits: int
    expired_assignments: int
    rejected_intents: int
    stale_assignment_rejections: int
    command_count: int
    distance_travelled: float
    recovery_delay_seconds: float | None


class MappingMissionSimulation:
    """Three-unit mission with deterministic allocation and explicit fault injection."""

    def __init__(self, output: Path, scenario: str = "unit_loss", *, speed: float = 8.0,
                 layout: MappingLayout | None = None, allocator: MappingAllocator | None = None) -> None:
        if scenario not in SCENARIOS:
            raise ValueError("Unknown mapping scenario")
        if not 0 < speed <= 20:
            raise ValueError("Mapping speed must be in (0, 20]")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose an empty output directory; existing mission evidence is preserved")
        output.mkdir(parents=True, exist_ok=True)
        self.output, self.scenario = output, scenario
        self.layout = layout or MappingLayout(speeds=(speed, speed, speed))
        self.allocator = allocator
        self.started_at = datetime.now(UTC)
        self.journal = JsonlAuditJournal(output / "coordinator.jsonl")
        layout_digest = sha256(json.dumps(asdict(self.layout), sort_keys=True).encode()).hexdigest()
        self.mission = Mission(uuid5(NAMESPACE_URL, f"aethelred-mapping-demo/v1:{layout_digest}"), 1,
                               self.started_at - timedelta(seconds=1),
                               self.started_at + timedelta(minutes=10),
                               frozenset({MissionCapability.MAP}),
                               frozenset({"mapper-1", "mapper-2", "mapper-3"}),
                               OperatingArea(Position(0, 0), Position(self.layout.width, self.layout.height)),
                               frozenset({"mapping-planner"}))
        self.coordinator = MappingCoordinator(self.mission, self.journal, allocator=allocator)
        self.coordinator.create_grid(self.layout.columns, self.layout.rows)
        self.journal.record("mapping_layout", str(self.mission.mission_id), {
            "layout": asdict(self.layout), "layout_sha256": layout_digest,
            "allocator_id": self.coordinator.allocator.allocator_id,
            "experimental_model_sha256": self.coordinator.allocator.artifact_sha256,
        })
        safety = OperationalSafetySupervisor()
        configuration = {
            "max_state_age_seconds": safety.max_state_age.total_seconds(),
            "max_sensor_age_seconds": safety.max_sensor_age.total_seconds(),
            "min_battery_reserve": safety.min_battery_reserve,
            "min_localisation_quality": safety.min_localisation_quality,
            "max_observation_uncertainty": safety.max_observation_uncertainty,
            "layout": asdict(self.layout),
            "allocator_id": self.coordinator.allocator.allocator_id,
            "experimental_model_sha256": self.coordinator.allocator.artifact_sha256,
        }
        policy_digest = sha256(Path(__file__).read_bytes()).hexdigest()
        self.units = []
        for index, (position, unit_speed) in enumerate(zip(self.layout.positions, self.layout.speeds, strict=True), start=1):
            vehicle_id = f"mapper-{index}"
            journal = JsonlAuditJournal(output / f"{vehicle_id}.jsonl")
            registry = MissionRegistry(journal)
            registry.register(self.mission, "demo-operator", "Bounded simulated mapping")
            configs = RuntimeConfigurationRegistry(journal)
            config = configs.register(configuration, "demo-operator", "Mapping demonstration limits")
            configs.activate(config.configuration_id, "demo-operator", "Reviewed demo configuration")
            lifecycle = RuntimeLifecycleSupervisor(journal)
            lifecycle.complete_self_test(("mapping-adapter", "intent-integrity", "safety"))
            lifecycle.load_mission(self.mission)
            lifecycle.arm("demo-operator")
            lifecycle.activate()
            health = RuntimeHealthSupervisor(journal, lifecycle, ("estimator", "adapter", "planner"))
            authenticator = IntentAuthenticator(secrets.token_bytes(32), journal)
            identity = RuntimeIdentity(self.mission.mission_id, f"mapping-demo:{policy_digest}",
                                       self.coordinator.allocator.artifact_sha256 or policy_digest, config.sha256)
            # Instantiate every safety parameter from the registered configuration.
            values = configs.active().values
            supervisor = OperationalSafetySupervisor(
                max_state_age=timedelta(seconds=_number(values["max_state_age_seconds"])),
                max_sensor_age=timedelta(seconds=_number(values["max_sensor_age_seconds"])),
                min_battery_reserve=_number(values["min_battery_reserve"]),
                min_localisation_quality=_number(values["min_localisation_quality"]),
                max_observation_uncertainty=_number(values["max_observation_uncertainty"]),
            )
            loop = AuthenticatedOperationalControlLoop(
                OperationalControlLoop(supervisor, journal, identity), authenticator,
                registry, lifecycle, configs, health)
            self.units.append(MappingUnit(MappingAdapter(vehicle_id, position, journal,
                                                        self.started_at, unit_speed), loop, authenticator, health))

    def run(self, max_ticks: int = 180) -> MappingResult:
        if not 1 <= max_ticks <= 300:
            raise ValueError("Simulation tick budget must be in [1, 300]")
        rejected = stale_rejections = visits = commands = 0
        samples: set[tuple[str, int]] = set()
        distance_travelled = 0.0
        stale_token = None
        recovery_delay = None
        for tick in range(max_ticks):
            now = self.started_at + timedelta(seconds=tick)
            if tick == 2 and self.scenario in {"unit_loss", "comms_loss"}:
                self.units[0].online = False
                owned = next((t for t in self.coordinator.tasks if t.vehicle_id == "mapper-1"
                              and t.status is TaskStatus.RUNNING), None)
                stale_token = owned
                self.journal.record("mapping_fault_injected", str(self.mission.mission_id),
                                    {"scenario": self.scenario, "tick": tick, "vehicle_id": "mapper-1"})
            if tick == 3 and self.scenario == "coordinator_restart":
                self.coordinator = MappingCoordinator(self.mission, self.journal, allocator=self.allocator)
                self.journal.record("mapping_coordinator_restarted", str(self.mission.mission_id), {"tick": tick})
            if tick == 8 and self.scenario == "comms_loss":
                self.units[0].online = True
                if stale_token is not None:
                    try:
                        self.coordinator.acknowledge(stale_token.task_id, "mapper-1",
                                                     stale_token.assignment_version, now)
                    except AssignmentError:
                        stale_rejections += 1
                        self.journal.record("mapping_stale_assignment_rejected", str(stale_token.task_id),
                                            {"tick": tick, "assignment_version": stale_token.assignment_version})
            self.coordinator.allocate(tuple(MappingVehicle(u.adapter.vehicle_id, u.adapter.position,
                                                           u.adapter.battery, u.online, u.adapter.speed)
                                                   for u in self.units), now)
            if stale_token is not None and recovery_delay is None:
                current = next(t for t in self.coordinator.tasks if t.task_id == stale_token.task_id)
                if current.assignment_version > stale_token.assignment_version:
                    recovery_delay = float(tick - 2)
            for unit in self.units:
                if not unit.online:
                    continue
                task = next((t for t in self.coordinator.tasks if t.vehicle_id == unit.adapter.vehicle_id
                             and t.status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}), None)
                if task is None:
                    continue
                self.coordinator.acknowledge(task.task_id, unit.adapter.vehicle_id, task.assignment_version, now)
                self.coordinator.heartbeat(task.task_id, unit.adapter.vehicle_id, task.assignment_version, now)
                unit.adapter.now = now
                for component in unit.health.required_component_ids:
                    unit.health.report(RuntimeHealthReport(component, True, now))
                sensor_time = now - timedelta(seconds=5) if (
                    self.scenario == "stale_sensor" and unit is self.units[0] and 2 <= tick < 7) else now
                state = WorldState(tick + 1, now, unit.adapter.vehicle_id, unit.adapter.position,
                                   True, True, unit.adapter.battery, 1.0, sensor_time, True, True, True,
                                   ObservationProvenance(uuid4(), "mapping-estimator", "local-enu",
                                                         "mapping-world/v1", tick, 0.0))
                proposal = IntentProposal(uuid4(), "deterministic-mapping-planner", self.mission.mission_id,
                                          self.mission.revision, state.revision, state.vehicle_id,
                                          MissionCapability.MAP, task.target, now + timedelta(seconds=2))
                unit.adapter.journal.record("mapping_assignment_intent", str(proposal.proposal_id), {
                    "task_id": str(task.task_id), "assignment_version": task.assignment_version,
                    "sample_index": task.completed_samples, "tick": tick,
                })
                before = unit.adapter.position
                try:
                    receipt = unit.loop.submit(unit.authenticator.sign(proposal, "mapping-planner", now),
                                               state, self.mission, unit.adapter, now)
                except PermissionError as error:
                    rejected += 1
                    unit.adapter.journal.record("mapping_intent_rejected", str(proposal.proposal_id),
                                                {"reason": str(error), "tick": tick})
                    continue
                commands += 1
                distance_travelled += before.distance_to(unit.adapter.position)
                if unit.adapter.position.distance_to(task.target) < 1e-8:
                    visits += 1
                    samples.add((str(task.task_id), task.completed_samples))
                    self.journal.record("mapping_sample_visited", str(receipt.command_id), {
                        "task_id": str(task.task_id), "sample_index": task.completed_samples,
                        "vehicle_id": state.vehicle_id, "assignment_version": task.assignment_version,
                        "tick": tick, "x": unit.adapter.position.x, "y": unit.adapter.position.y,
                    })
                    self.coordinator.record_sample(task.task_id, state.vehicle_id,
                                                   task.assignment_version, task.completed_samples, now)
            if self.coordinator.complete:
                break
        events = self.journal.read_all()
        result = MappingResult(self.scenario, self.coordinator.complete, tick + 1,
                               sum(t.status is TaskStatus.COMPLETED for t in self.coordinator.tasks),
                               len(self.coordinator.tasks), len(samples),
                               sum(len(t.samples) for t in self.coordinator.tasks), visits - len(samples),
                               sum(e["event_type"] == "mapping_task_expired" for e in events),
                               rejected, stale_rejections, commands, round(distance_travelled, 3), recovery_delay)
        (self.output / "summary.json").write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
        replay = {path.stem: JsonlAuditJournal(path).read_all()
                  for path in sorted(self.output.glob("*.jsonl"))}
        (self.output / "replay.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
        _write_replay(self.output / "replay.html", result, replay)
        return result


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Mapping configuration limits must be numeric")
    return float(value)  # type: ignore[arg-type]


def _write_replay(path: Path, result: MappingResult, replay: dict[str, list[dict]]) -> None:
    """Render verified journals in recording order, retaining every event payload."""
    events = sorted(((event["occurred_at"], source, event)
                     for source, entries in replay.items() for event in entries), key=lambda row: row[0])
    rows = "\n".join(
        f"<tr><td>{escape(timestamp)}</td><td>{escape(source)}</td>"
        f"<td>{escape(event['event_type'])}</td><td><details><summary>"
        f"{escape(event['correlation_id'])}</summary><pre>"
        f"{escape(json.dumps(event['payload'], indent=2))}</pre></details></td></tr>"
        for timestamp, source, event in events)
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Aethelred mapping replay</title><style>
body{{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px;background:#fafafa;color:#17202a}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ddd;padding:10px;text-align:left;vertical-align:top}}
pre{{white-space:pre-wrap;max-width:620px}}th{{background:#eaf2f8}}summary{{cursor:pointer}}
</style><h1>Aethelred mapping mission: {escape(result.scenario)}</h1>
<p>Completed: {result.completed}. Samples: {result.visited_samples}/{result.total_samples}.
Ticks: {result.ticks}. Expired assignments: {result.expired_assignments}.
Duplicate visits: {result.duplicate_visits}. Rejected intents: {result.rejected_intents}.</p>
<p>Abstract simulation of mapping sample visitation. Expand an event to inspect its evidence.
Rows follow recording time; simulated time and tick appear in the relevant payloads.</p>
<table><thead><tr><th>Recorded time</th><th>Source</th><th>Event</th><th>Evidence</th></tr></thead>
<tbody>{rows}</tbody></table></html>"""
    path.write_text(document, encoding="utf-8")
