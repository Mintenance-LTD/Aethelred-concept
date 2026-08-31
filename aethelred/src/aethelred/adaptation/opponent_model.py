"""Opponent behavior prediction using pattern matching and simple neural prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

import torch
from torch import nn

from aethelred.core.events import LossEvent
from aethelred.core.models import Vec2


@dataclass
class PredictedThreatAction:
    """A predicted future action of a threat entity."""

    threat_id: UUID
    predicted_position: Vec2
    predicted_action: str  # "advance", "retreat", "hold", "flank"
    confidence: float
    time_horizon: int  # steps ahead


@dataclass
class OpponentPattern:
    """A recognized pattern in opponent behavior."""

    pattern_type: str
    positions: list[Vec2] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0


class OpponentBehaviorPredictor(nn.Module):
    """Small LSTM for predicting opponent next positions."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(sequence)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


class OpponentBehaviorModel:
    """Phase 5: Predicts opponent next actions from observed movement patterns."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.predictor = OpponentBehaviorPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.movement_histories: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        self.patterns: list[OpponentPattern] = []
        self._kill_positions: list[Vec2] = []

    def observe_threat_positions(
        self, threat_positions: dict[UUID, Vec2], threat_velocities: dict[UUID, Vec2]
    ) -> None:
        for tid, pos in threat_positions.items():
            vel = threat_velocities.get(tid, Vec2())
            self.movement_histories[str(tid)].append(
                (pos.x / 1000.0, pos.y / 1000.0, vel.x / 20.0, vel.y / 20.0)
            )
            if len(self.movement_histories[str(tid)]) > 100:
                self.movement_histories[str(tid)] = self.movement_histories[str(tid)][-100:]

    def update_from_losses(self, losses: list[LossEvent]) -> None:
        for loss in losses:
            if loss.killing_threat_position_x is not None:
                self._kill_positions.append(Vec2(
                    x=loss.killing_threat_position_x,
                    y=loss.killing_threat_position_y or 0.0,
                ))

    def train_step(self) -> float:
        sequences = []
        targets = []

        for history in self.movement_histories.values():
            if len(history) < 10:
                continue
            for i in range(len(history) - 9):
                seq = history[i : i + 8]
                next_pos = history[i + 8]
                prev_pos = history[i + 7]
                dx = next_pos[0] - prev_pos[0]
                dy = next_pos[1] - prev_pos[1]
                sequences.append(seq)
                targets.append([dx, dy, 0.0, 0.5])

        if not sequences:
            return 0.0

        seq_tensor = torch.tensor(sequences, dtype=torch.float32, device=self.device)
        tgt_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self.predictor.train()
        predictions = self.predictor(seq_tensor)
        loss = nn.functional.mse_loss(predictions, tgt_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict_next_actions(
        self,
        current_positions: dict[UUID, Vec2],
        current_velocities: dict[UUID, Vec2],
    ) -> list[PredictedThreatAction]:
        self.predictor.eval()
        predictions = []

        for tid, pos in current_positions.items():
            history = self.movement_histories.get(str(tid), [])
            if len(history) < 8:
                vel = current_velocities.get(tid, Vec2())
                predictions.append(PredictedThreatAction(
                    threat_id=tid,
                    predicted_position=Vec2(x=pos.x + vel.x * 5, y=pos.y + vel.y * 5),
                    predicted_action="advance",
                    confidence=0.2,
                    time_horizon=5,
                ))
                continue

            seq = history[-8:]
            seq_tensor = torch.tensor([seq], dtype=torch.float32, device=self.device)
            pred = self.predictor(seq_tensor).squeeze(0)

            dx = pred[0].item() * 1000.0
            dy = pred[1].item() * 1000.0
            confidence = min(1.0, max(0.0, pred[3].item()))

            predicted_pos = Vec2(x=pos.x + dx, y=pos.y + dy)

            speed = (dx**2 + dy**2) ** 0.5
            if speed < 5.0:
                action = "hold"
            elif dx < 0:
                action = "retreat"
            else:
                action = "advance"

            predictions.append(PredictedThreatAction(
                threat_id=tid,
                predicted_position=predicted_pos,
                predicted_action=action,
                confidence=confidence,
                time_horizon=5,
            ))

        return predictions
