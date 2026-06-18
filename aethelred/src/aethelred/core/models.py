"""Core data models for the Aethelred system."""

from __future__ import annotations

import math
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field

from aethelred.core.enums import (
    DroneRole,
    DroneStatus,
    ThreatCategory,
    ThreatType,
)


class Vec2(BaseModel):
    """2D vector for positions and velocities."""

    x: float = 0.0
    y: float = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, other: Vec2) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: Vec2) -> float:
        return math.atan2(other.y - self.y, other.x - self.x)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalized(self) -> Vec2:
        mag = self.magnitude()
        if mag < 1e-8:
            return Vec2(x=0.0, y=0.0)
        return Vec2(x=self.x / mag, y=self.y / mag)

    def scaled(self, factor: float) -> Vec2:
        return Vec2(x=self.x * factor, y=self.y * factor)

    def __add__(self, other: Vec2) -> Vec2:  # type: ignore[override]
        return Vec2(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(x=self.x - other.x, y=self.y - other.y)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def rotated(self, angle: float) -> Vec2:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            x=self.x * cos_a - self.y * sin_a,
            y=self.x * sin_a + self.y * cos_a,
        )

    @staticmethod
    def from_angle(angle: float, magnitude: float = 1.0) -> Vec2:
        return Vec2(x=math.cos(angle) * magnitude, y=math.sin(angle) * magnitude)

    @staticmethod
    def lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
        return Vec2(x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t)


class DroneState(BaseModel):
    """Complete state of a single drone unit."""

    id: UUID = Field(default_factory=uuid4)
    role: DroneRole
    status: DroneStatus = DroneStatus.ACTIVE
    position: Vec2
    velocity: Vec2 = Field(default_factory=Vec2)
    heading: float = 0.0
    health: float = Field(default=1.0, ge=0.0, le=1.0)
    fuel: float = Field(default=1.0, ge=0.0, le=1.0)
    ammo: float = Field(default=1.0, ge=0.0, le=1.0)
    sensor_range: float = 100.0
    comm_range: float = 200.0
    max_speed: float = 20.0
    is_jammed: bool = False

    def is_alive(self) -> bool:
        return self.status not in (DroneStatus.DESTROYED,)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network input."""
        role_onehot = [0.0] * len(DroneRole)
        role_onehot[list(DroneRole).index(self.role)] = 1.0
        return np.array(
            [
                self.position.x,
                self.position.y,
                self.velocity.x,
                self.velocity.y,
                math.sin(self.heading),
                math.cos(self.heading),
                self.health,
                self.fuel,
                self.ammo,
                self.sensor_range / 200.0,
                float(self.is_jammed),
            ]
            + role_onehot,
            dtype=np.float32,
        )

    @staticmethod
    def feature_dim() -> int:
        return 11 + len(DroneRole)


class ThreatState(BaseModel):
    """State of a detected or known threat entity."""

    id: UUID = Field(default_factory=uuid4)
    threat_type: ThreatType
    category: ThreatCategory
    position: Vec2
    velocity: Vec2 = Field(default_factory=Vec2)
    heading: float = 0.0
    estimated_range: float = 50.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_active: bool = True
    first_seen_step: int = 0
    engagement_count: int = 0
    health: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network input."""
        type_onehot = [0.0] * len(ThreatType)
        type_onehot[list(ThreatType).index(self.threat_type)] = 1.0
        cat_onehot = [0.0] * len(ThreatCategory)
        cat_onehot[list(ThreatCategory).index(self.category)] = 1.0
        return np.array(
            [
                self.position.x,
                self.position.y,
                self.velocity.x,
                self.velocity.y,
                math.sin(self.heading),
                math.cos(self.heading),
                self.estimated_range / 100.0,
                self.confidence,
            ]
            + type_onehot
            + cat_onehot,
            dtype=np.float32,
        )

    @staticmethod
    def feature_dim() -> int:
        return 8 + len(ThreatType) + len(ThreatCategory)


class TerrainCell(BaseModel):
    """Single cell in the terrain grid."""

    position: Vec2
    elevation: float = 0.0
    cover_value: float = Field(default=0.0, ge=0.0, le=1.0)
    traversable: bool = True
    terrain_type: str = "open"


class ObjectiveState(BaseModel):
    """A mission objective."""

    id: UUID = Field(default_factory=uuid4)
    position: Vec2
    priority: float = Field(default=1.0, ge=0.0, le=1.0)
    is_completed: bool = False
    description: str = ""

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.position.x,
                self.position.y,
                self.priority,
                float(self.is_completed),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_dim() -> int:
        return 4


class BattlefieldState(BaseModel):
    """Complete snapshot of the battlefield at a single timestep."""

    timestep: int = 0
    friendly_units: list[DroneState] = Field(default_factory=list)
    threats: list[ThreatState] = Field(default_factory=list)
    objectives: list[ObjectiveState] = Field(default_factory=list)
    terrain_grid: Optional[np.ndarray] = Field(default=None)
    global_comms_degradation: float = Field(default=0.0, ge=0.0, le=1.0)
    weather_visibility: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def active_friendlies(self) -> list[DroneState]:
        return [u for u in self.friendly_units if u.is_alive()]

    @property
    def active_threats(self) -> list[ThreatState]:
        return [t for t in self.threats if t.is_active]

    @property
    def num_alive(self) -> int:
        return len(self.active_friendlies)

    @property
    def num_total(self) -> int:
        return len(self.friendly_units)

    def get_unit_by_id(self, unit_id: UUID) -> Optional[DroneState]:
        for u in self.friendly_units:
            if u.id == unit_id:
                return u
        return None

    def get_threat_by_id(self, threat_id: UUID) -> Optional[ThreatState]:
        for t in self.threats:
            if t.id == threat_id:
                return t
        return None
