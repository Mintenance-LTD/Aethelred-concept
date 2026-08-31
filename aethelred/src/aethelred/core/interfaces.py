"""Abstract interfaces for system components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from aethelred.core.actions import TacticalAction, TacticalDecision
from aethelred.core.events import LossEvent
from aethelred.core.models import BattlefieldState, DroneState, ThreatState


class TacticalAIInterface(ABC):
    """Contract for any tactical AI implementation."""

    @abstractmethod
    def decide(
        self, state: BattlefieldState, history: list[BattlefieldState]
    ) -> TacticalDecision: ...

    @abstractmethod
    def encode_state(self, state: BattlefieldState) -> torch.Tensor: ...

    @abstractmethod
    def get_policy_weights(self) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def load_policy_weights(self, weights: dict[str, torch.Tensor]) -> None: ...


class AdaptationInterface(ABC):
    """Contract for adaptation engines."""

    @abstractmethod
    def analyze_losses(self, losses: list[LossEvent]) -> dict[str, Any]: ...

    @abstractmethod
    def adapt(
        self, model: nn.Module, loss_analysis: dict[str, Any]
    ) -> nn.Module: ...

    @abstractmethod
    def get_model_delta(
        self, original: dict[str, torch.Tensor], updated: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: ...


class SwarmUnitInterface(ABC):
    """Contract for swarm unit behavior."""

    @abstractmethod
    def receive_policy_update(self, delta: dict[str, torch.Tensor]) -> None: ...

    @abstractmethod
    def act(
        self,
        local_observation: DroneState,
        nearby_threats: list[ThreatState],
    ) -> TacticalAction: ...

    @abstractmethod
    def report_telemetry(self) -> DroneState: ...


class SimulationInterface(ABC):
    """Contract for simulation environments."""

    @abstractmethod
    def reset(self, scenario: dict[str, Any] | None = None) -> BattlefieldState: ...

    @abstractmethod
    def step(
        self, decision: TacticalDecision
    ) -> tuple[BattlefieldState, float, bool, dict[str, Any]]: ...

    @abstractmethod
    def get_losses_since(self, timestep: int) -> list[LossEvent]: ...
