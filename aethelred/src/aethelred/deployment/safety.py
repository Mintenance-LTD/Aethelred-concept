"""Safety systems for real-world drone deployment.

Implements:
  - Geofencing (hard operational area boundaries)
  - Watchdog timer (detect stuck AI loops)
  - Sensor noise injection (training robustness)
  - Return-to-launch (RTL) failsafe
  - Action validation (sanity check AI outputs)
  - Heartbeat monitoring (detect comm loss)
  - Emergency stop protocol
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.enums import TacticalActionType, ThreatType
from aethelred.core.models import BattlefieldState, DroneState, ThreatState, Vec2

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """System safety state."""
    NOMINAL = "nominal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyConfig:
    """Configuration for safety systems."""
    # Geofence
    geofence_margin: float = 50.0  # meters from AO boundary
    geofence_hard_limit: float = 20.0  # absolute boundary

    # Watchdog
    watchdog_timeout_ms: float = 200.0  # max time for one decision cycle
    watchdog_max_failures: int = 3  # failures before RTL

    # Action validation
    max_speed_command: float = 30.0  # m/s max commanded speed
    max_acceleration: float = 15.0  # m/s^2
    min_altitude: float = 5.0  # meters (for future 3D)
    max_altitude: float = 120.0  # meters

    # Comms
    heartbeat_interval_s: float = 1.0
    heartbeat_timeout_s: float = 5.0

    # RTL
    rtl_fuel_threshold: float = 0.15  # trigger RTL below this fuel level
    rtl_health_threshold: float = 0.2  # trigger RTL below this health

    # Sensor noise (for training robustness)
    position_noise_std: float = 2.0  # meters
    velocity_noise_std: float = 0.5  # m/s
    heading_noise_std: float = 0.05  # radians
    sensor_dropout_prob: float = 0.02  # probability of sensor dropout per step

    # Simple Domain (survival floor for unmastered threats)
    simple_domain_enabled: bool = False
    simple_domain_mastery_threshold: float = 0.5
    simple_domain_buffer: float = 1.15  # push units to buffer * kill_range


@dataclass
class SafetyEvent:
    """Record of a safety system intervention."""
    timestamp: float
    level: SafetyLevel
    system: str
    message: str
    action_taken: str


class GeofenceGuard:
    """Ensures all commands stay within the operational area."""

    def __init__(
        self,
        width: float,
        height: float,
        margin: float = 50.0,
        hard_limit: float = 20.0,
    ) -> None:
        self.width = width
        self.height = height
        self.margin = margin
        self.hard_limit = hard_limit

    def validate_position(self, pos: Vec2) -> tuple[Vec2, bool]:
        """Validate and clamp a position to the geofence.

        Returns:
            (clamped_position, was_violated)
        """
        violated = False
        x, y = pos.x, pos.y

        if x < self.hard_limit or x > self.width - self.hard_limit:
            violated = True
            x = max(self.hard_limit, min(self.width - self.hard_limit, x))

        if y < self.hard_limit or y > self.height - self.hard_limit:
            violated = True
            y = max(self.hard_limit, min(self.height - self.hard_limit, y))

        return Vec2(x=x, y=y), violated

    def is_in_caution_zone(self, pos: Vec2) -> bool:
        """Check if position is in the soft margin zone."""
        return (
            pos.x < self.margin or pos.x > self.width - self.margin or
            pos.y < self.margin or pos.y > self.height - self.margin
        )

    def validate_decision(self, decision: TacticalDecision) -> TacticalDecision:
        """Validate all target positions in a decision."""
        for action in decision.actions:
            if action.target_position is not None:
                clamped, violated = self.validate_position(action.target_position)
                if violated:
                    logger.warning(
                        f"Geofence violation: {action.target_position} -> {clamped}"
                    )
                    action.target_position = clamped
        return decision


class WatchdogTimer:
    """Detects stuck or too-slow AI decision loops."""

    def __init__(self, timeout_ms: float = 200.0, max_failures: int = 3) -> None:
        self.timeout_ms = timeout_ms
        self.max_failures = max_failures
        self._start_time: Optional[float] = None
        self._consecutive_failures = 0
        self._total_timeouts = 0

    def start(self) -> None:
        """Start timing a decision cycle."""
        self._start_time = time.perf_counter()

    def check(self) -> tuple[bool, float]:
        """Check if the current cycle has exceeded the timeout.

        Returns:
            (is_ok, elapsed_ms)
        """
        if self._start_time is None:
            return True, 0.0

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0

        if elapsed_ms > self.timeout_ms:
            self._consecutive_failures += 1
            self._total_timeouts += 1
            logger.warning(
                f"Watchdog timeout: {elapsed_ms:.1f}ms > {self.timeout_ms}ms "
                f"(consecutive: {self._consecutive_failures})"
            )
            return False, elapsed_ms

        self._consecutive_failures = 0
        return True, elapsed_ms

    def should_rtl(self) -> bool:
        """Check if too many consecutive timeouts warrant RTL."""
        return self._consecutive_failures >= self.max_failures

    def reset(self) -> None:
        self._start_time = None

    @property
    def total_timeouts(self) -> int:
        return self._total_timeouts


class ActionValidator:
    """Validates AI-generated actions for physical plausibility."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def validate(
        self,
        action: TacticalAction,
        drone_state: DroneState,
    ) -> tuple[TacticalAction, list[str]]:
        """Validate an action and return corrected version + list of warnings."""
        warnings = []

        # Check target position is reasonable. Reach is bounded by the smaller of
        # the drone's own top speed and the configured commanded-speed cap.
        # (max_acceleration / min_altitude / max_altitude are reserved for the
        # future 3D model and intentionally not enforced in this 2D sim.)
        if action.target_position is not None:
            commanded_speed = min(drone_state.max_speed, self.config.max_speed_command)
            dist = drone_state.position.distance_to(action.target_position)
            max_range = commanded_speed * 10.0  # 10 seconds of travel
            if dist > max_range:
                # Scale down to reachable position
                direction = action.target_position - drone_state.position
                action.target_position = drone_state.position + direction.normalized().scaled(max_range)
                warnings.append(f"Target too far ({dist:.0f}m), clamped to {max_range:.0f}m")

        # Check fuel for non-RTL actions
        if drone_state.fuel < self.config.rtl_fuel_threshold:
            if action.action_type not in (TacticalActionType.RETREAT, TacticalActionType.HOLD):
                action.action_type = TacticalActionType.RETREAT
                warnings.append(f"Low fuel ({drone_state.fuel:.2f}), forced RETREAT")

        # Check health for non-evasive actions
        if drone_state.health < self.config.rtl_health_threshold:
            if action.action_type == TacticalActionType.ENGAGE:
                action.action_type = TacticalActionType.EVADE
                warnings.append(f"Low health ({drone_state.health:.2f}), forced EVADE")

        return action, warnings


class SensorNoiseInjector:
    """Injects realistic sensor noise for training robustness.

    When training in simulation, inject noise to ensure the AI learns to
    handle real-world sensor imperfections.
    """

    def __init__(self, config: SafetyConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng or np.random.default_rng()

    def inject_state_noise(self, state: BattlefieldState) -> BattlefieldState:
        """Add sensor noise to a battlefield state observation.

        Operates on a deep copy so the simulation's ground-truth state is never
        corrupted (the AI only ever sees the noisy observation, not the engine).
        """
        state = state.model_copy(deep=True)
        # Noise on friendly positions and velocities
        for drone in state.friendly_units:
            if not drone.is_alive():
                continue

            # Position noise (GPS-like)
            drone.position = Vec2(
                x=drone.position.x + float(self.rng.normal(0, self.config.position_noise_std)),
                y=drone.position.y + float(self.rng.normal(0, self.config.position_noise_std)),
            )

            # Velocity noise (IMU-like)
            drone.velocity = Vec2(
                x=drone.velocity.x + float(self.rng.normal(0, self.config.velocity_noise_std)),
                y=drone.velocity.y + float(self.rng.normal(0, self.config.velocity_noise_std)),
            )

            # Heading noise (compass-like)
            drone.heading += float(self.rng.normal(0, self.config.heading_noise_std))

        # Noise on threat detections (radar/camera noise)
        for threat in state.threats:
            if not threat.is_active:
                continue

            # Position noise (bigger for threats — detected at range)
            threat.position = Vec2(
                x=threat.position.x + float(self.rng.normal(0, self.config.position_noise_std * 2)),
                y=threat.position.y + float(self.rng.normal(0, self.config.position_noise_std * 2)),
            )

            # Sensor dropout: occasionally miss a threat
            if self.rng.random() < self.config.sensor_dropout_prob:
                threat.confidence *= 0.3  # drastically reduce confidence

        return state


class HeartbeatMonitor:
    """Monitors communication heartbeat between mother and swarm units."""

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self._last_heartbeat: dict[str, float] = {}

    def record_heartbeat(self, unit_id: str) -> None:
        self._last_heartbeat[unit_id] = time.time()

    def check_all(self) -> dict[str, bool]:
        """Check all units. Returns {unit_id: is_alive}."""
        now = time.time()
        results = {}
        for unit_id, last_time in self._last_heartbeat.items():
            results[unit_id] = (now - last_time) < self.timeout_s
        return results

    def get_lost_units(self) -> list[str]:
        """Get list of units that have lost heartbeat."""
        now = time.time()
        return [
            uid for uid, t in self._last_heartbeat.items()
            if (now - t) >= self.timeout_s
        ]


class SimpleDomain:
    """A minimal, hard-coded survival envelope — the tactical analogue of a JJK
    Simple Domain.

    It does not learn. It just refuses the sure-hit effect of a threat the swarm
    has not yet mastered: while a threat signature's mastery is below threshold,
    any unit that drifts inside that threat's estimated kill range is reflexively
    pushed back out (EVADE), regardless of the commanded action. This is shielded
    reinforcement learning — the policy may explore freely, but exploration
    against an unmastered lethal threat can never collect a free kill on our units,
    which keeps exploration non-suicidal long enough for the learner to adapt.

    :meth:`shield` is a pure function of the drone, the visible threats, and a
    mastery lookup, so it is trivially testable and has no hidden state.
    """

    def __init__(self, mastery_threshold: float = 0.5, buffer: float = 1.15) -> None:
        self.mastery_threshold = mastery_threshold
        self.buffer = buffer

    def shield(
        self,
        drone: DroneState,
        threats: list[ThreatState],
        mastery_of: Callable[[ThreatType], float],
    ) -> Optional[TacticalAction]:
        """Return an EVADE override if ``drone`` sits inside the kill range of an
        unmastered threat, else ``None`` (let the commanded action stand).

        When several unmastered threats endanger the unit, it evades the nearest
        one — the most immediate lethality.
        """
        if not drone.is_alive() or not threats:
            return None

        endangering: Optional[ThreatState] = None
        nearest = float("inf")
        for t in threats:
            if mastery_of(t.threat_type) >= self.mastery_threshold:
                continue  # mastered -> the policy is trusted to handle it
            dist = drone.position.distance_to(t.position)
            if dist <= t.estimated_range and dist < nearest:
                nearest = dist
                endangering = t

        if endangering is None:
            return None

        away = (drone.position - endangering.position).normalized()
        if away.magnitude() < 1e-6:
            away = Vec2(x=1.0, y=0.0)  # degenerate overlap -> pick an arbitrary bearing
        target = drone.position + away.scaled(endangering.estimated_range * self.buffer)
        return TacticalAction(
            action_type=TacticalActionType.EVADE,
            target_unit_id=drone.id,
            target_position=target,
            priority=1.0,
        )


class SafetyManager:
    """
    Central safety system that wraps around the tactical AI.
    Every decision passes through safety checks before execution.
    This is the LAST gate before commands reach the real hardware.
    """

    def __init__(
        self,
        config: SafetyConfig,
        battlefield_width: float = 1000.0,
        battlefield_height: float = 1000.0,
    ) -> None:
        self.config = config
        self.level = SafetyLevel.NOMINAL
        self.events: list[SafetyEvent] = []

        self.geofence = GeofenceGuard(
            width=battlefield_width,
            height=battlefield_height,
            margin=config.geofence_margin,
            hard_limit=config.geofence_hard_limit,
        )
        self.watchdog = WatchdogTimer(
            timeout_ms=config.watchdog_timeout_ms,
            max_failures=config.watchdog_max_failures,
        )
        self.action_validator = ActionValidator(config)
        self.heartbeat = HeartbeatMonitor(timeout_s=config.heartbeat_timeout_s)
        self.noise_injector = SensorNoiseInjector(config)
        self.simple_domain = SimpleDomain(
            mastery_threshold=config.simple_domain_mastery_threshold,
            buffer=config.simple_domain_buffer,
        )

        self._rtl_active = False
        self._emergency_stop = False

    def pre_decision(self, state: BattlefieldState) -> BattlefieldState:
        """Called before the AI makes a decision. Starts watchdog, updates safety level."""
        self.watchdog.start()
        self._update_safety_level(state)
        return state

    def post_decision(
        self,
        decision: TacticalDecision,
        state: BattlefieldState,
    ) -> TacticalDecision:
        """Called after the AI makes a decision. Validates and corrects."""
        # Check watchdog
        ok, elapsed_ms = self.watchdog.check()
        if not ok:
            self._record_event(
                SafetyLevel.WARNING, "watchdog",
                f"Decision took {elapsed_ms:.1f}ms (limit: {self.config.watchdog_timeout_ms}ms)",
                "logged_warning",
            )

        if self.watchdog.should_rtl():
            self._activate_rtl("Watchdog timeout threshold exceeded")
            return self._create_rtl_decision(state)

        if self._emergency_stop:
            return self._create_hold_decision(state)

        if self._rtl_active:
            return self._create_rtl_decision(state)

        # Geofence check
        decision = self.geofence.validate_decision(decision)

        # Per-action validation
        for i, action in enumerate(decision.actions):
            drone = self._find_drone(action.target_unit_id, state)
            if drone:
                validated, warnings = self.action_validator.validate(action, drone)
                decision.actions[i] = validated
                for w in warnings:
                    self._record_event(
                        SafetyLevel.CAUTION, "action_validator", w, "action_corrected"
                    )

        self.watchdog.reset()
        return decision

    def inject_training_noise(self, state: BattlefieldState) -> BattlefieldState:
        """Inject sensor noise for training robustness."""
        return self.noise_injector.inject_state_noise(state)

    def emergency_stop(self) -> None:
        """Activate emergency stop — all units hold position."""
        self._emergency_stop = True
        self._record_event(
            SafetyLevel.EMERGENCY, "operator",
            "Emergency stop activated", "all_units_hold",
        )
        logger.critical("EMERGENCY STOP ACTIVATED")

    def clear_emergency(self) -> None:
        """Clear emergency stop."""
        self._emergency_stop = False
        self._rtl_active = False
        self.level = SafetyLevel.NOMINAL
        logger.info("Emergency cleared, returning to nominal")

    def get_status(self) -> dict:
        """Get current safety system status."""
        return {
            "level": self.level.value,
            "rtl_active": self._rtl_active,
            "emergency_stop": self._emergency_stop,
            "watchdog_timeouts": self.watchdog.total_timeouts,
            "events_count": len(self.events),
            "lost_units": self.heartbeat.get_lost_units(),
        }

    def _update_safety_level(self, state: BattlefieldState) -> None:
        """Update global safety level based on current state."""
        alive = len(state.active_friendlies)
        total = max(len(state.friendly_units), 1)
        survival_rate = alive / total

        if survival_rate < 0.2:
            self.level = SafetyLevel.CRITICAL
        elif survival_rate < 0.4:
            self.level = SafetyLevel.WARNING
        elif survival_rate < 0.6:
            self.level = SafetyLevel.CAUTION
        else:
            self.level = SafetyLevel.NOMINAL

    def _activate_rtl(self, reason: str) -> None:
        """Activate return-to-launch for all units."""
        self._rtl_active = True
        self._record_event(
            SafetyLevel.CRITICAL, "safety_manager",
            f"RTL activated: {reason}", "rtl_all_units",
        )
        logger.warning(f"RTL ACTIVATED: {reason}")

    def _create_rtl_decision(self, state: BattlefieldState) -> TacticalDecision:
        """Create a decision that brings all units back to launch point."""
        actions = []
        launch_point = Vec2(x=100.0, y=state.terrain_grid.shape[0] * 10 if state.terrain_grid is not None else 500.0)
        for drone in state.active_friendlies:
            actions.append(TacticalAction(
                action_type=TacticalActionType.RETREAT,
                target_unit_id=drone.id,
                target_position=launch_point,
                priority=1.0,
            ))
        return TacticalDecision(
            timestep=state.timestep,
            actions=actions,
            confidence=1.0,
        )

    def _create_hold_decision(self, state: BattlefieldState) -> TacticalDecision:
        """Create a decision where all units hold position."""
        actions = []
        for drone in state.active_friendlies:
            actions.append(TacticalAction(
                action_type=TacticalActionType.HOLD,
                target_unit_id=drone.id,
                priority=1.0,
            ))
        return TacticalDecision(
            timestep=state.timestep,
            actions=actions,
            confidence=1.0,
        )

    def _find_drone(self, unit_id, state: BattlefieldState) -> Optional[DroneState]:
        if unit_id is None:
            return None
        for drone in state.friendly_units:
            if drone.id == unit_id:
                return drone
        return None

    def _record_event(
        self, level: SafetyLevel, system: str, message: str, action: str
    ) -> None:
        self.events.append(SafetyEvent(
            timestamp=time.time(),
            level=level,
            system=system,
            message=message,
            action_taken=action,
        ))
