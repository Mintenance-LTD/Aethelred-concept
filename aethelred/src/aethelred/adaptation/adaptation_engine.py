"""Adaptation Engine: orchestrates MAML, EWC, replay buffer, and threat classification."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aethelred.adaptation.ewc import EWCRegularizer
from aethelred.adaptation.maml import MAMLAdapter
from aethelred.adaptation.replay_buffer import Experience, PrioritizedReplayBuffer
from aethelred.adaptation.threat_classifier import HierarchicalThreatClassifier
from aethelred.config.settings import AdaptationConfig
from aethelred.core.enums import EngagementOutcome, TacticalActionType, ThreatType
from aethelred.core.events import EngagementEvent, LossEvent

# Index of each action type in the head's output space (matches enum order).
_ACTION_INDEX = {a: i for i, a in enumerate(TacticalActionType)}


@dataclass
class AdaptationResult:
    """Result of an adaptation cycle."""

    updated_weights: dict[str, torch.Tensor]
    method: str
    loss_before: float
    loss_after: float
    new_threat_registered: bool


class ThreatResponseHead(nn.Module):
    """Small network that maps loss features to corrective action logits.
    Used as a proxy for adaptation — gradients flow to the shared action head."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_actions: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptationEngine:
    """
    The Mahoraga system: orchestrates MAML + EWC + replay buffer + threat classification.
    Uses a small ThreatResponseHead for fast adaptation rather than passing data
    through the full DecisionTransformer.
    """

    def __init__(self, config: AdaptationConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.maml = MAMLAdapter(config.maml)
        self.ewc = EWCRegularizer(config.ewc)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_buffer_capacity,
            alpha=config.replay_alpha,
        )
        self.threat_classifier = HierarchicalThreatClassifier()
        self.known_threat_types: set[ThreatType] = set()
        self._adaptation_count = 0

        # Dedicated adaptation head for fast learning from losses
        self.threat_response_head = ThreatResponseHead().to(self.device)

    def adapt(
        self,
        model: nn.Module,
        losses: list[LossEvent],
        learning_rate: float = 1e-3,
        num_steps: int = 10,
    ) -> AdaptationResult:
        """
        Main adaptation logic:
        1. Classify threats from losses
        2. Train the threat response head on loss data
        3. Transfer learning signal to the model's action head weights
        """
        new_threats = set()
        for loss in losses:
            self.threat_classifier.update_from_loss(loss)
            if loss.cause_of_loss not in self.known_threat_types:
                new_threats.add(loss.cause_of_loss)
                self.known_threat_types.add(loss.cause_of_loss)

        has_new = len(new_threats) > 0
        support_data = self._losses_to_training_data(losses)

        if not support_data:
            return AdaptationResult(
                updated_weights={n: p.clone() for n, p in model.named_parameters()},
                method="no_data",
                loss_before=0.0,
                loss_after=0.0,
                new_threat_registered=has_new,
            )

        # Store these losses in the replay buffer and rehearse a few past ones so
        # adapting to a new threat does not wash out earlier counters (replay is
        # the second half of the catastrophic-forgetting defence alongside EWC).
        self._store_in_replay(losses, support_data)
        support_data = support_data + self._replay_rehearsal(exclude=len(support_data))

        # Compute loss before adaptation
        loss_before = self._compute_adaptation_loss(support_data)

        # Train the threat response head
        if has_new and self.config.enable_maml:
            adapted_head = self.maml.inner_loop_adapt(
                self.threat_response_head, support_data, F.cross_entropy
            )
            method = "maml_inner_loop"
        else:
            adapted_head = self._gradient_adapt_head(support_data, learning_rate, num_steps)
            method = "ewc_gradient" if self.ewc.num_tasks > 0 else "gradient"

        # Update the main threat response head
        self.threat_response_head.load_state_dict(adapted_head.state_dict())

        loss_after = self._compute_adaptation_loss(support_data)

        # Consolidate: once we have learned to counter a newly-seen threat,
        # register a Fisher-weighted EWC task so future adaptations preserve it.
        if has_new and self.config.enable_ewc:
            task_name = "counter_" + "_".join(sorted(t.value for t in new_threats))
            self.ewc.register_task(
                task_name, self.threat_response_head,
                iter(support_data), F.cross_entropy,
            )

        # Transfer adaptation signal to the transformer's action head:
        # Slightly adjust the model's action head weights based on the threat response head's
        # learned patterns. This creates a gradient-free transfer of the adaptation.
        adapted_model = copy.deepcopy(model)
        self._transfer_adaptation_to_model(adapted_model, adapted_head)

        updated_weights = {
            name: param.data.clone()
            for name, param in adapted_model.named_parameters()
        }

        self._adaptation_count += 1

        return AdaptationResult(
            updated_weights=updated_weights,
            method=method,
            loss_before=loss_before,
            loss_after=loss_after,
            new_threat_registered=has_new,
        )

    def register_mastery(self, task_name: str, model: nn.Module, data: list) -> None:
        """Register that a threat type has been mastered (for EWC)."""
        if not self.config.enable_ewc or not data:
            return
        self.ewc.register_task(
            task_name, self.threat_response_head, iter(data), F.cross_entropy
        )

    def get_model_delta(
        self,
        original: dict[str, torch.Tensor],
        updated: dict[str, torch.Tensor],
        threshold: float = 1e-5,
    ) -> dict[str, torch.Tensor]:
        """Compute compressed delta between original and updated weights."""
        delta = {}
        for name, original_weight in original.items():
            if name in updated:
                diff = updated[name] - original_weight
                mask = diff.abs() > threshold
                if mask.any():
                    delta[name] = diff * mask.float()
        return delta

    def get_threat_summary(self) -> dict:
        return self.threat_classifier.get_summary()

    def record_neutralizations(self, engagements: list[EngagementEvent]) -> None:
        """Update threat knowledge when the swarm destroys a threat.

        Feeds the classifier's exchange-ratio / mastery tracking, which in turn
        drives counter-effectiveness metrics and priority targeting.
        """
        for e in engagements:
            if e.outcome == EngagementOutcome.TARGET_DESTROYED and e.threat_type is not None:
                self.threat_classifier.record_neutralization(e.threat_type)

    def _transfer_adaptation_to_model(
        self, model: nn.Module, adapted_head: nn.Module
    ) -> None:
        """Transfer learning signal from the adapted threat response head to the
        transformer's action_head. We do this by slightly biasing the action head's
        output layer toward the adapted head's learned biases."""
        # Get the adapted head's final layer bias (learned evasion preference)
        adapted_bias = None
        for name, param in adapted_head.named_parameters():
            if "bias" in name and param.shape[0] == 8:  # final layer bias
                adapted_bias = param.data.clone()
                break

        if adapted_bias is None:
            return

        # Apply a small bias adjustment to the model's action type head
        for name, param in model.named_parameters():
            if "action_type_head" in name and "bias" in name:
                if param.shape == adapted_bias.shape:
                    # Blend: 90% original + 10% adapted
                    param.data = param.data * 0.9 + adapted_bias * 0.1
                break

    def _gradient_adapt_head(
        self,
        training_data: list[tuple[torch.Tensor, torch.Tensor]],
        learning_rate: float,
        num_steps: int,
    ) -> nn.Module:
        """Standard gradient update on the threat response head."""
        adapted = copy.deepcopy(self.threat_response_head)
        adapted.train()
        optimizer = torch.optim.Adam(adapted.parameters(), lr=learning_rate)

        for step in range(num_steps):
            total_loss = torch.tensor(0.0, device=self.device)

            for inputs, targets in training_data:
                outputs = adapted(inputs)
                loss = F.cross_entropy(outputs, targets)
                total_loss = total_loss + loss

            if self.config.enable_ewc and self.ewc.num_tasks > 0:
                ewc_penalty = self.ewc.penalty(adapted)
                total_loss = total_loss + ewc_penalty

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
            optimizer.step()

        return adapted

    def _losses_to_training_data(
        self, losses: list[LossEvent]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Convert loss events into (features, counter-action) training pairs.

        The target is the *recommended counter* for the threat/geometry that
        caused the loss — not a constant — so the head learns a genuine
        threat->response mapping rather than a single fixed action.
        """
        data = []
        for loss in losses:
            features = torch.tensor([
                loss.lost_unit_position_x / 1000.0,
                loss.lost_unit_position_y / 1000.0,
                loss.final_velocity_x / 20.0,
                loss.final_velocity_y / 20.0,
                loss.engagement_range / 100.0,
                float(loss.cause_of_loss == ThreatType.SMALL_ARMS),
                float(loss.cause_of_loss == ThreatType.AA_MISSILE),
                float(loss.cause_of_loss == ThreatType.EW_JAMMING),
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            target = torch.tensor(
                [self._counter_action_index(loss)], dtype=torch.long, device=self.device
            )
            data.append((features, target))

        return data

    @staticmethod
    def _counter_action_index(loss: LossEvent) -> int:
        """Map a loss to the recommended counter-action's head index."""
        cause = loss.cause_of_loss
        rng = loss.engagement_range
        if cause in (ThreatType.EW_JAMMING, ThreatType.EW_SPOOFING):
            action = TacticalActionType.FORMATION_CHANGE  # spread / re-route mesh
        elif cause == ThreatType.LASER:
            action = TacticalActionType.RETREAT            # break line of sight
        elif 0.0 < rng < 30.0:
            action = TacticalActionType.RETREAT            # close ambush -> withdraw
        else:
            action = TacticalActionType.EVADE              # kinetic at range -> jink
        return _ACTION_INDEX[action]

    def _store_in_replay(
        self,
        losses: list[LossEvent],
        support_data: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Persist loss experiences (high priority) for later rehearsal."""
        for loss, (features, target) in zip(losses, support_data):
            feat_np = features.detach().cpu().numpy().reshape(-1)
            self.replay_buffer.add(Experience(
                state_features=feat_np,
                action_features=target.detach().cpu().numpy().reshape(-1),
                reward=-1.0,
                next_state_features=feat_np,
                done=True,
                loss_event=loss,
                threat_type=loss.cause_of_loss,
                timestep=loss.timestep,
            ))

    def _replay_rehearsal(
        self, exclude: int, k: int = 8
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Sample past experiences to interleave with current adaptation data."""
        if len(self.replay_buffer) <= exclude:
            return []
        experiences, _, _ = self.replay_buffer.sample(min(k, len(self.replay_buffer)))
        pairs = []
        for exp in experiences:
            feat = torch.tensor(
                exp.state_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            tgt = torch.tensor(
                [int(exp.action_features[0])], dtype=torch.long, device=self.device
            )
            pairs.append((feat, tgt))
        return pairs

    def _compute_adaptation_loss(
        self, data: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Compute loss on the threat response head."""
        if not data:
            return 0.0
        total = 0.0
        self.threat_response_head.eval()
        with torch.no_grad():
            for inputs, targets in data:
                outputs = self.threat_response_head(inputs)
                loss = F.cross_entropy(outputs, targets)
                total += loss.item()
        return total / len(data)
