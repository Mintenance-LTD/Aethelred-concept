"""Mother Drone: the command node running the full tactical AI."""

from __future__ import annotations

import logging

from aethelred.core.actions import TacticalDecision
from aethelred.core.enums import DroneRole
from aethelred.core.events import LossEvent
from aethelred.core.models import BattlefieldState, DroneState, Vec2
from aethelred.learning.learning_loop import LearningLoop
from aethelred.swarm.coordinator import SwarmCoordinator
from aethelred.tactical_ai.policy import TacticalPolicy

logger = logging.getLogger(__name__)


class MotherDrone:
    """
    Command node. Runs the full tactical AI, aggregates telemetry,
    manages the learning loop, and commands the swarm.
    """

    def __init__(
        self,
        tactical_policy: TacticalPolicy,
        learning_loop: LearningLoop,
        swarm_coordinator: SwarmCoordinator,
        position: Vec2 | None = None,
    ) -> None:
        self.tactical_policy = tactical_policy
        self.learning_loop = learning_loop
        self.coordinator = swarm_coordinator

        # Mother drone's own state (high altitude, persistent overwatch)
        self.state = DroneState(
            role=DroneRole.MOTHER,
            position=position or Vec2(x=300.0, y=500.0),
            sensor_range=500.0,
            comm_range=500.0,
            max_speed=5.0,  # slow but persistent
        )

        # LearningLoop produces offline-only candidates. It is never connected
        # to the coordinator because policy changes require independent approval.

    def tick(self, battlefield_state: BattlefieldState) -> TacticalDecision:
        """
        One command cycle:
        1. Aggregate telemetry into BattlefieldState
        2. Run tactical AI
        3. Decompose into per-unit commands
        4. Feed learning loop
        5. Check if adaptation should trigger
        6. Return the tactical decision
        """
        # 1. Update mother position in coordinator
        self.coordinator.set_mother_position(self.state.position)

        # 2. Update communication status (mutates per-unit comm state)
        self.coordinator.update_comm_status()

        # 3. Detect threats (update confidence based on sensor range)
        for threat in battlefield_state.active_threats:
            dist = self.state.position.distance_to(threat.position)
            if dist < self.state.sensor_range:
                threat.confidence = min(1.0, threat.confidence + 0.1)

        # 4. Run tactical AI
        decision = self.tactical_policy.decide(battlefield_state)

        # 5. Assign tasks to swarm units
        self.coordinator.assign_tasks(decision, battlefield_state.active_threats)

        # 6. Feed learning loop (Phase 1: Observation)
        reward = self._estimate_step_reward(battlefield_state)
        self.learning_loop.on_step(battlefield_state, reward)

        # 7. Check for adaptation (Phases 2-5)
        adaptation = self.learning_loop.maybe_adapt()
        if adaptation:
            logger.info(
                f"[MOTHER] Adaptation event: method={adaptation.adaptation_method}, "
                f"threats={[t.value for t in adaptation.threat_types_addressed]}"
            )

        return decision

    def handle_unit_loss(self, loss_event: LossEvent) -> None:
        """Forward unit loss to learning loop and remove from coordinator."""
        self.learning_loop.on_loss(loss_event)
        self.coordinator.remove_unit(loss_event.lost_unit_id)
        logger.info(
            f"[MOTHER] Lost unit {loss_event.lost_unit_role} "
            f"to {loss_event.cause_of_loss.value}"
        )

    def get_stats(self) -> dict:
        """Get current mother drone and system stats."""
        return {
            "mother_position": (self.state.position.x, self.state.position.y),
            "swarm_active": self.coordinator.active_count,
            "swarm_total": self.coordinator.unit_count,
            "learning_loop": self.learning_loop.stats,
        }

    def _estimate_step_reward(self, state: BattlefieldState) -> float:
        """Quick estimate of step quality for learning loop."""
        alive_ratio = len(state.active_friendlies) / max(len(state.friendly_units), 1)
        threats_active = len(state.active_threats)
        return alive_ratio - threats_active * 0.05
