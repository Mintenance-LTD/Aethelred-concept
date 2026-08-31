"""Non-offensive adapter between operational commands and the simulator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.enums import TacticalActionType
from aethelred.core.models import BattlefieldState
from aethelred.runtime.operational import AuthorisedCommand, CommandReceipt, MissionCapability


class DecisionSimulation(Protocol):
    """The decision-only portion of the simulator required by this adapter."""

    def get_current_state(self) -> BattlefieldState | None: ...

    def step_decision(
        self, decision: TacticalDecision
    ) -> tuple[object, float, bool, bool, dict[str, object]]: ...


class SimulatorCommandAdapter:
    """Execute allowed operational capabilities through one simulator decision."""

    def __init__(self, simulator: DecisionSimulation) -> None:
        self._simulator = simulator

    def execute(self, command: AuthorisedCommand) -> CommandReceipt:
        """Translate an authorised non-offensive command into simulator actions."""
        state = self._simulator.get_current_state()
        if state is None:
            return CommandReceipt(
                command_id=command.command_id,
                accepted=False,
                recorded_at=datetime.now(UTC),
                sequence=command.sequence,
                detail="Simulator has no current world state",
            )

        decision = TacticalDecision(
            timestep=state.timestep,
            actions=[
                TacticalAction(
                    action_type=self._action_type_for(command.capability),
                    target_unit_id=unit.id,
                    target_position=command.target_position or unit.position,
                    priority=1.0,
                )
                for unit in state.active_friendlies
            ],
            confidence=1.0,
        )
        self._simulator.step_decision(decision)
        return CommandReceipt(
            command_id=command.command_id,
            accepted=True,
            recorded_at=datetime.now(UTC),
            sequence=command.sequence,
            detail="Executed through simulator decision interface",
        )

    @staticmethod
    def _action_type_for(capability: MissionCapability) -> TacticalActionType:
        """Map only bounded capabilities; this adapter never emits ENGAGE."""
        mapping = {
            MissionCapability.HOLD: TacticalActionType.HOLD,
            MissionCapability.RETURN_HOME: TacticalActionType.RETREAT,
            MissionCapability.SURVEY: TacticalActionType.RECON,
            MissionCapability.INSPECT: TacticalActionType.RECON,
            MissionCapability.MAP: TacticalActionType.RECON,
            MissionCapability.RELAY: TacticalActionType.RELAY,
            MissionCapability.SEARCH: TacticalActionType.RECON,
        }
        return mapping[capability]
