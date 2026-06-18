"""Simple 2D physics for drone movement and engagement resolution."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from aethelred.config.settings import PhysicsConfig
from aethelred.core.enums import DroneStatus
from aethelred.core.models import DroneState, ThreatState, Vec2

if TYPE_CHECKING:
    from aethelred.simulation.battlefield import Battlefield


class SimplePhysics:
    """Kinematic movement, hit probability, attack resolution."""

    def __init__(self, config: PhysicsConfig) -> None:
        self.dt = config.dt
        self.hit_probability_base = config.hit_probability_base
        self.cover_damage_reduction = config.cover_damage_reduction
        self.fuel_consumption_rate = config.fuel_consumption_rate
        self.damage_per_hit = config.damage_per_hit

    def move_drone(
        self, drone: DroneState, target_pos: Vec2, battlefield: "Battlefield"
    ) -> DroneState:
        """Move a drone toward a target position."""
        if not drone.is_alive():
            return drone

        direction = target_pos - drone.position
        dist = direction.magnitude()

        if dist < 1.0:
            return drone

        move_dir = direction.normalized()
        speed = drone.max_speed * self.dt
        actual_move = min(speed, dist)

        new_pos = drone.position + move_dir.scaled(actual_move)
        new_pos = battlefield.clamp_to_bounds(new_pos)

        drone.position = new_pos
        drone.velocity = move_dir.scaled(drone.max_speed)
        drone.heading = math.atan2(move_dir.y, move_dir.x)
        drone.fuel = max(0.0, drone.fuel - self.fuel_consumption_rate * actual_move)

        if drone.fuel <= 0.0:
            drone.status = DroneStatus.RTB

        return drone

    def move_threat(
        self, threat: ThreatState, target_pos: Vec2, speed: float, battlefield: "Battlefield"
    ) -> ThreatState:
        """Move a threat entity toward a target."""
        if not threat.is_active:
            return threat

        direction = target_pos - threat.position
        dist = direction.magnitude()

        if dist < 1.0:
            return threat

        move_dir = direction.normalized()
        actual_move = min(speed * self.dt, dist)

        new_pos = threat.position + move_dir.scaled(actual_move)
        new_pos = battlefield.clamp_to_bounds(new_pos)

        threat.position = new_pos
        threat.velocity = move_dir.scaled(speed)
        threat.heading = math.atan2(move_dir.y, move_dir.x)

        return threat

    def compute_hit_probability(
        self,
        attacker_pos: Vec2,
        target_pos: Vec2,
        attacker_range: float,
        target_velocity: Vec2,
        battlefield: "Battlefield",
    ) -> float:
        """Compute probability of hitting a target."""
        dist = attacker_pos.distance_to(target_pos)

        # Range factor: degrades beyond optimal range
        if dist > attacker_range * 2:
            return 0.0
        range_factor = max(0.0, 1.0 - (dist / (attacker_range * 2)) ** 2)

        # Cover factor
        cover = battlefield.get_cover_at(target_pos)
        cover_factor = 1.0 - cover * self.cover_damage_reduction

        # Movement factor: harder to hit fast targets
        speed = target_velocity.magnitude()
        movement_factor = 1.0 / (1.0 + speed * 0.05)

        # LOS check
        if not battlefield.line_of_sight(attacker_pos, target_pos):
            return 0.0

        return self.hit_probability_base * range_factor * cover_factor * movement_factor

    def resolve_attack(
        self,
        attacker_pos: Vec2,
        target_pos: Vec2,
        attacker_range: float,
        target_velocity: Vec2,
        battlefield: "Battlefield",
    ) -> tuple[float, bool]:
        """Resolve an attack. Returns (damage_dealt, is_destroyed)."""
        hit_prob = self.compute_hit_probability(
            attacker_pos, target_pos, attacker_range, target_velocity, battlefield
        )

        if random.random() > hit_prob:
            return 0.0, False

        damage = self.damage_per_hit
        return damage, damage >= 1.0

    def apply_damage(self, drone: DroneState, damage: float) -> DroneState:
        """Apply damage to a drone."""
        drone.health = max(0.0, drone.health - damage)
        if drone.health <= 0.0:
            drone.status = DroneStatus.DESTROYED
        elif drone.health < 0.3:
            drone.status = DroneStatus.DAMAGED
        return drone

    def apply_threat_damage(self, threat: ThreatState, damage: float) -> ThreatState:
        """Apply damage to a threat."""
        threat.health = max(0.0, threat.health - damage)
        if threat.health <= 0.0:
            threat.is_active = False
        return threat
