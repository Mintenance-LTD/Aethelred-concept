"""Reward model for computing composite tactical rewards."""

from __future__ import annotations

from typing import Optional

from aethelred.core.actions import TacticalDecision
from aethelred.core.events import LossEvent
from aethelred.core.models import BattlefieldState


class RewardModel:
    """
    Computes composite reward signals for training.
    Combines: mission progress, survival, efficiency, threat neutralization,
    and adaptation bonus.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {
            "mission_progress": 1.0,
            "survival": 0.5,
            "efficiency": 0.3,
            "threat_neutralized": 0.8,
            "adaptation_bonus": 0.2,
        }

    def compute_reward(
        self,
        state_before: BattlefieldState,
        state_after: BattlefieldState,
        action: TacticalDecision,
        losses: list[LossEvent],
    ) -> float:
        reward = 0.0

        # Survival ratio
        total = max(len(state_before.friendly_units), 1)
        alive_after = len(state_after.active_friendlies)
        survival = alive_after / total
        reward += self.weights["survival"] * survival

        # Loss penalty
        reward -= len(losses) * 0.3

        # Threats neutralized
        threats_before = len(state_before.active_threats)
        threats_after = len(state_after.active_threats)
        neutralized = max(0, threats_before - threats_after)
        reward += self.weights["threat_neutralized"] * neutralized * 0.5

        # Mission progress
        completed_before = sum(1 for o in state_before.objectives if o.is_completed)
        completed_after = sum(1 for o in state_after.objectives if o.is_completed)
        new_completions = completed_after - completed_before
        reward += self.weights["mission_progress"] * new_completions

        # Efficiency: fewer actions with more impact is better
        if action.actions:
            engagement_actions = sum(
                1 for a in action.actions if a.action_type.value == "engage"
            )
            if engagement_actions > 0 and neutralized > 0:
                efficiency = neutralized / engagement_actions
                reward += self.weights["efficiency"] * efficiency * 0.2

        return reward

    def compute_return_to_go(
        self, future_rewards: list[float], gamma: float = 0.99
    ) -> list[float]:
        """Compute discounted return-to-go for Decision Transformer training."""
        rtg = [0.0] * len(future_rewards)
        running = 0.0
        for i in reversed(range(len(future_rewards))):
            running = future_rewards[i] + gamma * running
            rtg[i] = running
        return rtg
