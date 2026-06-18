"""Lightweight autonomous swarm unit agent."""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn

from aethelred.core.actions import TacticalAction
from aethelred.core.enums import DroneRole, TacticalActionType
from aethelred.core.interfaces import SwarmUnitInterface
from aethelred.core.models import DroneState, ThreatState, Vec2


class LightweightPolicy(nn.Module):
    """Small neural network for autonomous unit decision-making."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwarmUnit(SwarmUnitInterface):
    """
    Lightweight autonomous agent running a distilled policy.
    Falls back to local decision-making when communication is lost.
    """

    def __init__(self, drone_state: DroneState) -> None:
        self.state = drone_state
        self.local_policy = LightweightPolicy()
        self.last_command: Optional[dict] = None
        self.comm_status: str = "connected"  # connected, degraded, lost
        self.autonomy_level: float = 0.0  # 0=commanded, 1=autonomous

    def receive_command(self, command: dict) -> None:
        """Receive a command from the mother drone."""
        self.last_command = command
        self.autonomy_level = 0.0

    def receive_policy_update(self, delta: dict[str, torch.Tensor]) -> None:
        """Apply model weight delta to local policy."""
        current_state = self.local_policy.state_dict()
        for name, diff in delta.items():
            short_name = name.split(".")[-1] if "." in name else name
            # Try to match by suffix
            for key in current_state:
                if key.endswith(short_name) or key == name:
                    if current_state[key].shape == diff.shape:
                        current_state[key] = current_state[key] + diff
                    break
        self.local_policy.load_state_dict(current_state, strict=False)

    def act(
        self,
        local_observation: DroneState,
        nearby_threats: list[ThreatState],
    ) -> TacticalAction:
        """Choose an action based on comm status and available info."""
        self.state = local_observation

        if self.comm_status == "connected" and self.last_command:
            return self._execute_command()
        elif self.comm_status == "degraded":
            return self._blended_decision(nearby_threats)
        else:
            return self._autonomous_decision(nearby_threats)

    def report_telemetry(self) -> DroneState:
        return self.state.model_copy()

    def set_comm_status(self, status: str) -> None:
        self.comm_status = status
        if status == "lost":
            self.autonomy_level = 1.0
        elif status == "degraded":
            self.autonomy_level = 0.5

    def _execute_command(self) -> TacticalAction:
        """Execute the last received command."""
        cmd = self.last_command
        if cmd is None:
            return self._default_action()

        return TacticalAction(
            action_type=TacticalActionType(cmd.get("action_type", "hold")),
            target_unit_id=self.state.id,
            target_position=cmd.get("target_position"),
            target_threat_id=cmd.get("target_threat_id"),
            priority=cmd.get("priority", 0.5),
        )

    def _autonomous_decision(self, nearby_threats: list[ThreatState]) -> TacticalAction:
        """Fully autonomous decision using local policy."""
        if not nearby_threats:
            return self._role_default_action()

        # Find closest threat
        closest = min(
            nearby_threats,
            key=lambda t: self.state.position.distance_to(t.position),
        )
        dist = self.state.position.distance_to(closest.position)

        # Role-based autonomous behavior
        if self.state.role == DroneRole.RECON:
            # Recon: maintain safe distance, observe
            if dist < closest.estimated_range * 1.2:
                away = self.state.position - closest.position
                evade_pos = self.state.position + away.normalized().scaled(50.0)
                return TacticalAction(
                    action_type=TacticalActionType.EVADE,
                    target_unit_id=self.state.id,
                    target_position=evade_pos,
                    priority=0.8,
                )
            return TacticalAction(
                action_type=TacticalActionType.RECON,
                target_unit_id=self.state.id,
                target_position=closest.position,
                priority=0.5,
            )

        elif self.state.role == DroneRole.ENGAGE:
            # Engage: attack if in range, close if not
            if dist < self.state.sensor_range and self.state.ammo > 0:
                return TacticalAction(
                    action_type=TacticalActionType.ENGAGE,
                    target_unit_id=self.state.id,
                    target_threat_id=closest.id,
                    target_position=closest.position,
                    priority=0.9,
                )
            elif dist < self.state.sensor_range * 2:
                return TacticalAction(
                    action_type=TacticalActionType.MOVE,
                    target_unit_id=self.state.id,
                    target_position=closest.position,
                    priority=0.6,
                )

        elif self.state.role == DroneRole.EW:
            # EW: stay at medium range
            if dist < 50.0:
                away = self.state.position - closest.position
                return TacticalAction(
                    action_type=TacticalActionType.EVADE,
                    target_unit_id=self.state.id,
                    target_position=self.state.position + away.normalized().scaled(80.0),
                    priority=0.7,
                )

        return self._role_default_action()

    def _blended_decision(self, nearby_threats: list[ThreatState]) -> TacticalAction:
        """Blend command with autonomous decision based on autonomy level."""
        autonomous = self._autonomous_decision(nearby_threats)
        if self.last_command and random.random() > self.autonomy_level:
            return self._execute_command()
        return autonomous

    def _role_default_action(self) -> TacticalAction:
        """Default action based on role when no threats detected."""
        if self.state.role == DroneRole.RECON:
            # Patrol forward
            forward = Vec2.from_angle(self.state.heading, 50.0)
            return TacticalAction(
                action_type=TacticalActionType.RECON,
                target_unit_id=self.state.id,
                target_position=self.state.position + forward,
                priority=0.3,
            )
        elif self.state.role == DroneRole.RELAY:
            return TacticalAction(
                action_type=TacticalActionType.HOLD,
                target_unit_id=self.state.id,
                priority=0.2,
            )
        return self._default_action()

    def _default_action(self) -> TacticalAction:
        return TacticalAction(
            action_type=TacticalActionType.HOLD,
            target_unit_id=self.state.id,
            priority=0.1,
        )
