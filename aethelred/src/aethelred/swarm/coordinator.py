"""Swarm coordination: task decomposition, formation, and policy propagation."""

from __future__ import annotations

import logging
from uuid import UUID

import torch

from aethelred.core.actions import TacticalDecision
from aethelred.core.enums import DroneRole, FormationType, TacticalActionType
from aethelred.core.models import DroneState, ThreatState, Vec2
from aethelred.swarm.comms import CommChannel
from aethelred.swarm.formations import FormationManager
from aethelred.swarm.swarm_unit import SwarmUnit

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """
    Manages swarm formation, task assignment, and policy propagation.
    Translates high-level TacticalDecisions into per-unit commands.
    """

    def __init__(
        self,
        comm_range: float = 500.0,
        formation_spacing: float = 30.0,
    ) -> None:
        self.units: dict[UUID, SwarmUnit] = {}
        self.formation_manager = FormationManager()
        self.comm_channel = CommChannel(base_range=comm_range)
        self.formation_spacing = formation_spacing
        self._mother_position: Vec2 = Vec2(x=500.0, y=500.0)
        self._current_formation = FormationType.DIAMOND

    def register_unit(self, unit: SwarmUnit) -> None:
        self.units[unit.state.id] = unit

    def remove_unit(self, unit_id: UUID) -> None:
        self.units.pop(unit_id, None)

    def set_mother_position(self, pos: Vec2) -> None:
        self._mother_position = pos

    def assign_tasks(
        self,
        decision: TacticalDecision,
        threats: list[ThreatState],
    ) -> list[dict]:
        """
        Decompose a TacticalDecision into individual unit commands.
        Role-based assignment:
          - RECON: recon/move actions
          - ENGAGE: engage/evade actions
          - EW: electronic warfare actions
          - RELAY: maintain comm mesh positions
        """
        commands: list[dict[str, object]] = []
        cmd: dict[str, object]

        recon_units = [u for u in self.units.values() if u.state.role == DroneRole.RECON]
        engage_units = [u for u in self.units.values() if u.state.role == DroneRole.ENGAGE]
        ew_units = [u for u in self.units.values() if u.state.role == DroneRole.EW]
        relay_units = [u for u in self.units.values() if u.state.role == DroneRole.RELAY]

        # Global action from decision (simplified: use first action's type)
        global_action = TacticalActionType.HOLD
        target_pos = None
        formation = self._current_formation

        if decision.actions:
            act = decision.actions[0]
            global_action = act.action_type
            target_pos = act.target_position
            if act.formation:
                formation = act.formation
                self._current_formation = formation

        # Assign RECON units
        for unit in recon_units:
            cmd = {
                "action_type": "recon" if global_action != TacticalActionType.RETREAT else "retreat",
                "target_position": target_pos,
                "priority": 0.5,
            }
            commands.append(cmd)
            unit.receive_command(cmd)

        # Assign ENGAGE units
        for i, unit in enumerate(engage_units):
            if global_action == TacticalActionType.ENGAGE and threats:
                # Distribute across threats
                threat_idx = i % len(threats)
                cmd = {
                    "action_type": "engage",
                    "target_position": threats[threat_idx].position,
                    "target_threat_id": threats[threat_idx].id,
                    "priority": 0.9,
                }
            elif global_action == TacticalActionType.EVADE:
                cmd = {
                    "action_type": "evade",
                    "target_position": target_pos,
                    "priority": 0.8,
                }
            else:
                cmd = {
                    "action_type": global_action.value,
                    "target_position": target_pos,
                    "priority": 0.5,
                }
            commands.append(cmd)
            unit.receive_command(cmd)

        # EW units hold position or jam
        for unit in ew_units:
            cmd = {
                "action_type": "hold" if global_action != TacticalActionType.RETREAT else "retreat",
                "target_position": target_pos,
                "priority": 0.4,
            }
            commands.append(cmd)
            unit.receive_command(cmd)

        # RELAY units maintain mesh
        for unit in relay_units:
            cmd = {
                "action_type": "relay",
                "priority": 0.3,
            }
            commands.append(cmd)
            unit.receive_command(cmd)

        return commands

    def update_formations(
        self, formation: FormationType, center: Vec2
    ) -> dict[UUID, Vec2]:
        """Compute and return formation positions."""
        self._current_formation = formation
        unit_ids = list(self.units.keys())
        return self.formation_manager.compute_positions(
            formation, center, unit_ids, spacing=self.formation_spacing
        )

    def propagate_policy_update(self, delta: dict[str, torch.Tensor]) -> None:
        """Broadcast policy update to all connected units."""
        for unit_id, unit in self.units.items():
            if self.comm_channel.can_reach(self._mother_position, unit.state):
                unit.receive_policy_update(delta)
            else:
                logger.warning(f"Cannot reach unit {unit_id} for policy update")

    def update_comm_status(self) -> dict[UUID, str]:
        """Update communication status for all units."""
        statuses = {}
        for unit_id, unit in self.units.items():
            status = self.comm_channel.get_status(self._mother_position, unit.state)
            if status.signal_strength > 0.5:
                comm = "connected"
            elif status.signal_strength > 0.1:
                comm = "degraded"
            else:
                comm = "lost"
            unit.set_comm_status(comm)
            statuses[unit_id] = comm
        return statuses

    def get_swarm_telemetry(self) -> list[DroneState]:
        """Collect telemetry from all units."""
        return [unit.report_telemetry() for unit in self.units.values()]

    def get_active_units(self) -> list[SwarmUnit]:
        return [u for u in self.units.values() if u.state.is_alive()]

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def active_count(self) -> int:
        return len(self.get_active_units())
