"""Enemy threat behavior and spawning patterns."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from aethelred.config.settings import ThreatSpawnerConfig
from aethelred.core.enums import ThreatType, THREAT_TYPE_TO_CATEGORY
from aethelred.core.models import ThreatState, Vec2
from aethelred.simulation.battlefield import Battlefield
from aethelred.simulation.entity_manager import EntityManager

if TYPE_CHECKING:
    from aethelred.simulation.physics import SimplePhysics


@dataclass
class EnemyTacticProfile:
    """Defines a pattern of enemy behavior."""

    name: str = "patrol"
    threat_types: list[ThreatType] = field(
        default_factory=lambda: [ThreatType.SMALL_ARMS]
    )
    behavior: str = "patrol"  # static, patrol, ambush, pursue, retreat
    aggression: float = 0.5
    coordination: float = 0.3
    speed: float = 8.0
    engagement_range: float = 60.0
    spawn_pattern: str = "random"  # random, clustered, perimeter


TACTIC_PROFILES: dict[str, EnemyTacticProfile] = {
    "static": EnemyTacticProfile(
        name="static",
        threat_types=[ThreatType.SMALL_ARMS, ThreatType.AA_MISSILE],
        behavior="static",
        aggression=0.3,
        speed=0.0,
        engagement_range=80.0,
    ),
    "patrol": EnemyTacticProfile(
        name="patrol",
        threat_types=[ThreatType.SMALL_ARMS],
        behavior="patrol",
        aggression=0.5,
        speed=8.0,
        engagement_range=60.0,
    ),
    "ambush": EnemyTacticProfile(
        name="ambush",
        threat_types=[ThreatType.AA_MISSILE, ThreatType.SMALL_ARMS],
        behavior="ambush",
        aggression=0.9,
        speed=10.0,
        engagement_range=150.0,
        spawn_pattern="clustered",
    ),
    "ew_attack": EnemyTacticProfile(
        name="ew_attack",
        threat_types=[ThreatType.EW_JAMMING, ThreatType.EW_SPOOFING],
        behavior="patrol",
        aggression=0.4,
        speed=6.0,
        engagement_range=150.0,
    ),
    "aggressive": EnemyTacticProfile(
        name="aggressive",
        threat_types=[ThreatType.SMALL_ARMS, ThreatType.KINETIC_INTERCEPT, ThreatType.AA_MISSILE],
        behavior="pursue",
        aggression=1.0,
        speed=15.0,
        engagement_range=120.0,
        spawn_pattern="perimeter",
    ),
    # --- Experiment profiles: a winnable vs a deadly opposition. The optimal
    # response differs (engage the small-arms / withdraw from the long-range AA),
    # so a state-dependent policy can beat any single constant action. ---
    "exp_winnable": EnemyTacticProfile(
        name="exp_winnable",
        threat_types=[ThreatType.SMALL_ARMS],   # range 45 < engage sensor 80 -> safe kills
        behavior="static",
        aggression=0.6,
        speed=0.0,
        engagement_range=45.0,
        spawn_pattern="clustered",
    ),
    "exp_deadly": EnemyTacticProfile(
        name="exp_deadly",
        threat_types=[ThreatType.AA_MISSILE],   # range 140 >> engage 80 -> must withdraw
        behavior="static",
        aggression=0.9,
        speed=0.0,
        engagement_range=140.0,
        spawn_pattern="clustered",
    ),
}

# Per-episode opposition for the "random_mix" profile (state-dependent optimum).
RANDOM_MIX_POOL = ["exp_winnable", "exp_deadly"]


class ThreatSpawner:
    """Manages enemy threat spawning and behavior."""

    def __init__(self, config: ThreatSpawnerConfig) -> None:
        self.config = config
        self.active_profile = TACTIC_PROFILES.get(
            config.default_profile, TACTIC_PROFILES["patrol"]
        )
        self._rng: Optional[np.random.Generator] = None
        self._patrol_waypoints: dict[str, list[Vec2]] = {}  # threat_id -> waypoints
        self._patrol_indices: dict[str, int] = {}

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng or np.random.default_rng()
        self._patrol_waypoints.clear()
        self._patrol_indices.clear()
        # "random_mix" draws a fresh opposition each episode so the optimal
        # response is state-dependent rather than a single fixed action.
        if self.config.default_profile == "random_mix":
            choice = RANDOM_MIX_POOL[int(self._rng.integers(len(RANDOM_MIX_POOL)))]
            self.active_profile = TACTIC_PROFILES[choice]

    def spawn_initial_threats(
        self, entity_manager: EntityManager, battlefield: Battlefield, timestep: int = 0
    ) -> list[ThreatState]:
        """Spawn the initial set of threats."""
        spawned = []
        for i in range(self.config.initial_threats):
            threat = self._create_threat(battlefield, timestep)
            entity_manager.add_threat(threat)
            spawned.append(threat)
            self._setup_patrol(threat, battlefield)
        return spawned

    def maybe_spawn(
        self, timestep: int, entity_manager: EntityManager, battlefield: Battlefield
    ) -> list[ThreatState]:
        """Conditionally spawn new threats based on schedule."""
        active_count = len(entity_manager.get_active_threats())
        if active_count >= self.config.max_threats:
            return []
        if timestep > 0 and timestep % self.config.spawn_interval == 0:
            threat = self._create_threat(battlefield, timestep)
            entity_manager.add_threat(threat)
            self._setup_patrol(threat, battlefield)
            return [threat]
        return []

    def update_threat_behavior(
        self,
        entity_manager: EntityManager,
        battlefield: Battlefield,
        physics: "SimplePhysics",
    ) -> None:
        """Move threats according to their tactic profile."""
        active_drones = entity_manager.get_active_drones()

        for threat in entity_manager.get_active_threats():
            tid = str(threat.id)

            if self.active_profile.behavior == "static":
                continue

            elif self.active_profile.behavior == "pursue":
                # Chase closest drone
                if active_drones:
                    closest = min(
                        active_drones,
                        key=lambda d: d.position.distance_to(threat.position),
                    )
                    physics.move_threat(
                        threat, closest.position,
                        self.active_profile.speed, battlefield,
                    )

            elif self.active_profile.behavior == "patrol":
                # Follow waypoints
                waypoints = self._patrol_waypoints.get(tid, [])
                if not waypoints:
                    self._setup_patrol(threat, battlefield)
                    waypoints = self._patrol_waypoints.get(tid, [])
                if waypoints:
                    idx = self._patrol_indices.get(tid, 0)
                    target = waypoints[idx]
                    if threat.position.distance_to(target) < 10.0:
                        idx = (idx + 1) % len(waypoints)
                        self._patrol_indices[tid] = idx
                        target = waypoints[idx]
                    physics.move_threat(
                        threat, target,
                        self.active_profile.speed, battlefield,
                    )

            elif self.active_profile.behavior == "ambush":
                # Stay hidden until drone is close, then attack
                if active_drones:
                    closest = min(
                        active_drones,
                        key=lambda d: d.position.distance_to(threat.position),
                    )
                    dist = closest.position.distance_to(threat.position)
                    if dist < self.active_profile.engagement_range:
                        physics.move_threat(
                            threat, closest.position,
                            self.active_profile.speed * 1.5, battlefield,
                        )

    def set_profile(self, profile_name: str) -> None:
        """Switch the active tactic profile."""
        if profile_name in TACTIC_PROFILES:
            self.active_profile = TACTIC_PROFILES[profile_name]

    def _create_threat(self, battlefield: Battlefield, timestep: int) -> ThreatState:
        """Create a single threat entity."""
        rng = self._rng or np.random.default_rng()
        threat_type = random.choice(self.active_profile.threat_types)

        if self.active_profile.spawn_pattern == "perimeter":
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                pos = Vec2(x=float(rng.uniform(0, battlefield.width)), y=battlefield.height * 0.9)
            elif side == "bottom":
                pos = Vec2(x=float(rng.uniform(0, battlefield.width)), y=battlefield.height * 0.1)
            elif side == "left":
                pos = Vec2(x=battlefield.width * 0.1, y=float(rng.uniform(0, battlefield.height)))
            else:
                pos = Vec2(x=battlefield.width * 0.9, y=float(rng.uniform(0, battlefield.height)))
        elif self.active_profile.spawn_pattern == "clustered":
            # Cluster near drone approach path (center-right of map)
            center = Vec2(x=battlefield.width * 0.5, y=battlefield.height * 0.5)
            pos = Vec2(
                x=center.x + float(rng.uniform(-120, 120)),
                y=center.y + float(rng.uniform(-120, 120)),
            )
        else:  # random
            pos = Vec2(
                x=float(rng.uniform(battlefield.width * 0.5, battlefield.width * 0.95)),
                y=float(rng.uniform(battlefield.height * 0.05, battlefield.height * 0.95)),
            )

        return ThreatState(
            threat_type=threat_type,
            category=THREAT_TYPE_TO_CATEGORY[threat_type],
            position=pos,
            estimated_range=self.active_profile.engagement_range,
            confidence=0.0,
            first_seen_step=timestep,
        )

    def _setup_patrol(self, threat: ThreatState, battlefield: Battlefield) -> None:
        """Generate patrol waypoints for a threat."""
        rng = self._rng or np.random.default_rng()
        tid = str(threat.id)
        num_waypoints = random.randint(3, 6)
        waypoints = []
        for _ in range(num_waypoints):
            wp = Vec2(
                x=float(rng.uniform(
                    max(0, threat.position.x - 150),
                    min(battlefield.width, threat.position.x + 150),
                )),
                y=float(rng.uniform(
                    max(0, threat.position.y - 150),
                    min(battlefield.height, threat.position.y + 150),
                )),
            )
            waypoints.append(wp)
        self._patrol_waypoints[tid] = waypoints
        self._patrol_indices[tid] = 0
