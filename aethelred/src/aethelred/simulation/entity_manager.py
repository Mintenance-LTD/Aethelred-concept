"""Entity lifecycle management for the simulation."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

import numpy as np

from aethelred.config.settings import SimulationConfig
from aethelred.core.enums import (
    DroneRole,
    EngagementOutcome,
    TacticalActionType,
)
from aethelred.core.actions import TacticalDecision
from aethelred.core.events import EngagementEvent, LossEvent
from aethelred.core.models import DroneState, ThreatState, Vec2
from aethelred.simulation.battlefield import Battlefield
from aethelred.simulation.physics import SimplePhysics


class EntityManager:
    """Manages all entities (drones and threats) in the simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.drones: dict[UUID, DroneState] = {}
        self.threats: dict[UUID, ThreatState] = {}
        self._loss_events: list[LossEvent] = []
        self._engagement_events: list[EngagementEvent] = []

    def reset(self, scenario: Optional[dict] = None, rng: Optional[np.random.Generator] = None) -> None:
        """Reset all entities. Spawn fresh swarm and threats."""
        self.drones.clear()
        self.threats.clear()
        self._loss_events.clear()
        self._engagement_events.clear()

        if rng is None:
            rng = np.random.default_rng()

        bf_w = self.config.battlefield.width
        bf_h = self.config.battlefield.height

        # Spawn swarm units
        center = Vec2(x=bf_w * 0.3, y=bf_h * 0.5)
        role_counts = [
            (DroneRole.RECON, self.config.swarm.num_recon),
            (DroneRole.ENGAGE, self.config.swarm.num_engage),
            (DroneRole.EW, self.config.swarm.num_ew),
            (DroneRole.RELAY, self.config.swarm.num_relay),
        ]

        role_configs = {
            DroneRole.RECON: {"sensor_range": 150.0, "max_speed": 25.0, "ammo": 0.0},
            DroneRole.ENGAGE: {"sensor_range": 80.0, "max_speed": 20.0, "ammo": 1.0},
            DroneRole.EW: {"sensor_range": 120.0, "max_speed": 15.0, "ammo": 0.0},
            DroneRole.RELAY: {"sensor_range": 50.0, "max_speed": 18.0, "comm_range": 400.0, "ammo": 0.0},
        }

        idx = 0
        for role, count in role_counts:
            for _ in range(count):
                offset = Vec2(
                    x=float(rng.uniform(-50, 50)),
                    y=float(rng.uniform(-50, 50)),
                )
                pos = center + offset
                rc = role_configs[role]
                drone = DroneState(
                    role=role,
                    position=pos,
                    sensor_range=rc.get("sensor_range", 100.0),
                    max_speed=rc.get("max_speed", 20.0),
                    ammo=rc.get("ammo", 1.0),
                    comm_range=rc.get("comm_range", 200.0),
                )
                self.drones[drone.id] = drone
                idx += 1

    def get_drone_list(self) -> list[DroneState]:
        return list(self.drones.values())

    def get_threat_list(self) -> list[ThreatState]:
        return list(self.threats.values())

    def get_active_drones(self) -> list[DroneState]:
        return [d for d in self.drones.values() if d.is_alive()]

    def get_active_threats(self) -> list[ThreatState]:
        return [t for t in self.threats.values() if t.is_active]

    def add_threat(self, threat: ThreatState) -> None:
        self.threats[threat.id] = threat

    def apply_decision(
        self, decision: TacticalDecision, physics: SimplePhysics, battlefield: Battlefield
    ) -> None:
        """Apply tactical decision: move drones according to actions."""
        for action in decision.actions:
            if action.target_unit_id is None:
                continue
            drone = self.drones.get(action.target_unit_id)
            if drone is None or not drone.is_alive():
                continue

            if action.action_type in (TacticalActionType.MOVE, TacticalActionType.RECON):
                if action.target_position is not None:
                    physics.move_drone(drone, action.target_position, battlefield)
            elif action.action_type == TacticalActionType.EVADE:
                if action.target_position is not None:
                    physics.move_drone(drone, action.target_position, battlefield)
            elif action.action_type == TacticalActionType.RETREAT:
                # Move away from center of threats
                active_threats = self.get_active_threats()
                if active_threats:
                    from aethelred.utils.geometry import compute_centroid
                    threat_center = compute_centroid([t.position for t in active_threats])
                    away = drone.position - threat_center
                    target = drone.position + away.normalized().scaled(100.0)
                    target = battlefield.clamp_to_bounds(target)
                    physics.move_drone(drone, target, battlefield)
            # HOLD, RELAY, FORMATION_CHANGE: no movement

    def resolve_engagements(
        self,
        decision: TacticalDecision,
        physics: SimplePhysics,
        battlefield: Battlefield,
        timestep: int,
    ) -> tuple[list[EngagementEvent], list[LossEvent]]:
        """Resolve all combat engagements for this timestep."""
        engagements: list[EngagementEvent] = []
        losses: list[LossEvent] = []

        # Friendly fire on threats (engagement actions)
        for action in decision.actions:
            if action.action_type != TacticalActionType.ENGAGE:
                continue
            if action.target_unit_id is None or action.target_threat_id is None:
                continue

            drone = self.drones.get(action.target_unit_id)
            threat = self.threats.get(action.target_threat_id)
            if drone is None or threat is None:
                continue
            if not drone.is_alive() or not threat.is_active:
                continue
            if drone.ammo <= 0.0:
                continue

            damage, destroyed = physics.resolve_attack(
                drone.position, threat.position, drone.sensor_range,
                threat.velocity, battlefield,
            )

            th_before = threat.health
            if damage > 0:
                physics.apply_threat_damage(threat, damage)
                drone.ammo = max(0.0, drone.ammo - 0.1)

            outcome = (
                EngagementOutcome.TARGET_DESTROYED if not threat.is_active
                else EngagementOutcome.TARGET_DAMAGED if damage > 0
                else EngagementOutcome.MISS
            )

            engagements.append(EngagementEvent(
                timestep=timestep,
                friendly_id=drone.id,
                threat_id=threat.id,
                threat_type=threat.threat_type,
                outcome=outcome,
                friendly_health_before=drone.health,
                friendly_health_after=drone.health,
                threat_health_before=th_before,
                threat_health_after=threat.health,
                distance=drone.position.distance_to(threat.position),
            ))

        # Threats fire on drones
        for threat in self.get_active_threats():
            # Find closest active drone in range
            closest_drone: Optional[DroneState] = None
            closest_dist = float("inf")
            for drone in self.get_active_drones():
                dist = drone.position.distance_to(threat.position)
                if dist < threat.estimated_range and dist < closest_dist:
                    closest_dist = dist
                    closest_drone = drone

            if closest_drone is None:
                continue

            damage, destroyed = physics.resolve_attack(
                threat.position, closest_drone.position, threat.estimated_range,
                closest_drone.velocity, battlefield,
            )

            dh_before = closest_drone.health
            if damage > 0:
                physics.apply_damage(closest_drone, damage)

            if not closest_drone.is_alive():
                # Unit destroyed — create LossEvent (high-value training data)
                loss = LossEvent(
                    timestep=timestep,
                    lost_unit_id=closest_drone.id,
                    lost_unit_role=closest_drone.role.value,
                    lost_unit_position_x=closest_drone.position.x,
                    lost_unit_position_y=closest_drone.position.y,
                    cause_of_loss=threat.threat_type,
                    killing_threat_id=threat.id,
                    killing_threat_position_x=threat.position.x,
                    killing_threat_position_y=threat.position.y,
                    engagement_range=closest_dist,
                    final_heading=closest_drone.heading,
                    final_velocity_x=closest_drone.velocity.x,
                    final_velocity_y=closest_drone.velocity.y,
                    num_friendlies_alive=len(self.get_active_drones()),
                    num_threats_active=len(self.get_active_threats()),
                )
                losses.append(loss)
                self._loss_events.append(loss)

            outcome = (
                EngagementOutcome.UNIT_LOST if not closest_drone.is_alive()
                else EngagementOutcome.TARGET_DAMAGED if damage > 0
                else EngagementOutcome.MISS
            )

            engagements.append(EngagementEvent(
                timestep=timestep,
                friendly_id=closest_drone.id,
                threat_id=threat.id,
                threat_type=threat.threat_type,
                outcome=outcome,
                friendly_health_before=dh_before,
                friendly_health_after=closest_drone.health,
                threat_health_before=threat.health,
                threat_health_after=threat.health,
                distance=closest_dist,
            ))

        self._engagement_events.extend(engagements)
        return engagements, losses

    def get_losses_since(self, timestep: int) -> list[LossEvent]:
        return [ev for ev in self._loss_events if ev.timestep >= timestep]

    def get_all_losses(self) -> list[LossEvent]:
        return list(self._loss_events)

    def get_all_engagements(self) -> list[EngagementEvent]:
        return list(self._engagement_events)
