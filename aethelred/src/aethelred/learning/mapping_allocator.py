"""Offline learned task-duration model and simulation-only allocation candidates."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from aethelred.runtime.geometry import Position
from aethelred.runtime.mapping_allocation import TaskAssignment
from aethelred.runtime.mapping_tasks import MappingVehicle, MissionTask
from aethelred.runtime.operational import OperatingArea

FEATURE_SCHEMA = "mapping-duration/v1:arrival_seconds,route_seconds,samples_remaining,intercept"


def task_features(vehicle: MappingVehicle, task: MissionTask) -> tuple[float, ...]:
    points = task.samples[task.completed_samples:]
    route = sum(a.distance_to(b) for a, b in itertools.pairwise(points))
    return (vehicle.position.distance_to(points[0]) / vehicle.speed,
            route / vehicle.speed, float(len(points)), 1.0)


def analytic_task_ticks(vehicle: MappingVehicle, task: MissionTask) -> float:
    points = (vehicle.position, *task.samples[task.completed_samples:])
    return float(sum(max(1, math.ceil(a.distance_to(b) / vehicle.speed))
                     for a, b in itertools.pairwise(points)))


@dataclass(frozen=True)
class DurationModel:
    coefficients: tuple[float, ...]
    training_seeds: tuple[int, ...]
    artifact_sha256: str

    def predict(self, vehicle: MappingVehicle, task: MissionTask) -> float:
        return max(1.0, sum(w * x for w, x in zip(self.coefficients, task_features(vehicle, task), strict=True)))

    @classmethod
    def load(cls, path: Path) -> DurationModel:
        encoded = path.read_bytes()
        raw = json.loads(encoded)
        if raw.get("feature_schema") != FEATURE_SCHEMA or raw.get("model_type") != "ridge-regression":
            raise ValueError("Unsupported mapping duration model")
        weights = tuple(float(v) for v in raw["coefficients"])
        if len(weights) != 4 or not all(math.isfinite(w) for w in weights):
            raise ValueError("Mapping duration model coefficients must be finite")
        seeds = tuple(raw["training_seeds"])
        if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
            raise ValueError("Invalid training split provenance")
        return cls(weights, seeds, sha256(encoded).hexdigest())


class BalancedTaskAllocator:
    """Propose a batch minimizing its maximum predicted task duration.

    The analytic variant is a stronger deterministic control for the learned
    variant. Both search exactly the same assignment combinations. Neither owns
    tasks or sends vehicle commands.
    """

    def __init__(self, model: DurationModel | None = None) -> None:
        self.model = model
        self.allocator_id = "learned-balanced/v1" if model else "analytic-balanced/v1"
        self.artifact_sha256 = model.artifact_sha256 if model else None

    def propose(self, vehicles: tuple[MappingVehicle, ...],
                tasks: tuple[MissionTask, ...]) -> tuple[TaskAssignment, ...]:
        count = min(len(vehicles), len(tasks))
        if count == 0:
            return ()
        if len(vehicles) > 3 or len(tasks) > 12:
            raise ValueError("Experimental batch allocation supports at most 3 units and 12 cells")
        costs = {(v.vehicle_id, t.task_id): (self.model.predict(v, t) if self.model
                                           else analytic_task_ticks(v, t)) for v in vehicles for t in tasks}
        best_key = None
        best: tuple[TaskAssignment, ...] = ()
        for units in itertools.combinations(sorted(vehicles, key=lambda v: v.vehicle_id), count):
            for routes in itertools.permutations(tasks, count):
                durations = tuple(costs[(v.vehicle_id, t.task_id)] for v, t in zip(units, routes, strict=True))
                key = (max(durations), sum(durations),
                       tuple((v.vehicle_id, str(t.task_id)) for v, t in zip(units, routes, strict=True)))
                if best_key is None or key < best_key:
                    best_key = key
                    best = tuple(TaskAssignment(v.vehicle_id, t.task_id) for v, t in zip(units, routes, strict=True))
        return best


def duration_dataset(seeds: tuple[int, ...], examples_per_seed: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic geometry labels from the documented discrete movement model."""
    from uuid import NAMESPACE_URL, uuid5
    features, targets = [], []
    for seed in seeds:
        rng = random.Random(seed)
        for index in range(examples_per_seed):
            x, y = rng.uniform(0, 60), rng.uniform(0, 40)
            width, height = rng.uniform(5, 35), rng.uniform(5, 30)
            samples = tuple(Position(x + dx * width, y + dy * height)
                            for dx, dy in ((.25, .25), (.75, .25), (.75, .75), (.25, .75)))
            task = MissionTask(uuid5(NAMESPACE_URL, f"training/{seed}/{index}"),
                               uuid5(NAMESPACE_URL, "mapping-training"), 1,
                               OperatingArea(Position(x, y), Position(x + width, y + height)),
                               samples, completed_samples=rng.randrange(4))
            vehicle = MappingVehicle("training-unit", Position(rng.uniform(0, 100), rng.uniform(0, 80)),
                                     .9, speed=rng.uniform(4, 12))
            features.append(task_features(vehicle, task))
            targets.append(analytic_task_ticks(vehicle, task))
    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def train_duration_model(path: Path, training_seeds: tuple[int, ...] = tuple(range(64)),
                         validation_seeds: tuple[int, ...] = tuple(range(500, 516))) -> dict:
    """Fit once offline with fixed hyperparameters; validation never updates weights."""
    if not training_seeds or not validation_seeds or set(training_seeds) & set(validation_seeds):
        raise ValueError("Training and validation seeds must be non-empty and disjoint")
    if path.exists():
        raise FileExistsError("Existing model evidence will not be overwritten")
    x, y = duration_dataset(training_seeds)
    regularisation = 1e-6
    weights = np.linalg.solve(x.T @ x + regularisation * np.eye(x.shape[1]), x.T @ y)
    vx, vy = duration_dataset(validation_seeds)
    record = {
        "model_type": "ridge-regression", "feature_schema": FEATURE_SCHEMA,
        "coefficients": weights.tolist(), "training_seeds": list(training_seeds),
        "validation_seeds": list(validation_seeds), "examples_per_seed": 40,
        "regularisation": regularisation,
        "target": "Exact task ticks under the abstract straight-line movement model",
        "training_mae_ticks": float(np.mean(np.abs(x @ weights - y))),
        "validation_mae_ticks": float(np.mean(np.abs(vx @ weights - vy))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
