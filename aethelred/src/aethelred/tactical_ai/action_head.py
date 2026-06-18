"""Action head: converts transformer hidden states into tactical actions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F



class ActionHead(nn.Module):
    """
    Hybrid discrete/continuous action head.
    Outputs:
        - action_type: categorical over TacticalActionType
        - target_position: 2D normalized coordinates
        - priority: scalar [0, 1]
        - formation: categorical over FormationType
        - target_logits: pointer over threats for target selection
    """

    def __init__(
        self,
        hidden_dim: int,
        num_action_types: int = 8,
        num_formations: int = 6,
        max_targets: int = 16,
    ) -> None:
        super().__init__()
        self.num_action_types = num_action_types
        self.num_formations = num_formations
        self.max_targets = max_targets

        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # normalized [0, 1] coordinates
        )
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.formation_head = nn.Linear(hidden_dim, num_formations)
        self.target_head = nn.Linear(hidden_dim, max_targets)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        Returns:
            dict of action logits/values
        """
        return {
            "action_type_logits": self.action_type_head(hidden),
            "target_position": self.position_head(hidden),
            "priority": self.priority_head(hidden),
            "formation_logits": self.formation_head(hidden),
            "target_logits": self.target_head(hidden),
        }

    def sample_action(
        self,
        logits: dict[str, torch.Tensor],
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample concrete actions from the output distributions."""
        # Action type
        at_logits = logits["action_type_logits"]
        if at_logits.dim() == 3:
            at_logits = at_logits[:, -1, :]  # last timestep
        if deterministic:
            action_type = at_logits.argmax(dim=-1)
        else:
            probs = F.softmax(at_logits / temperature, dim=-1)
            action_type = torch.multinomial(probs, 1).squeeze(-1)

        # Target position
        target_pos = logits["target_position"]
        if target_pos.dim() == 3:
            target_pos = target_pos[:, -1, :]

        # Priority
        priority = logits["priority"]
        if priority.dim() == 3:
            priority = priority[:, -1, :]

        # Formation
        fm_logits = logits["formation_logits"]
        if fm_logits.dim() == 3:
            fm_logits = fm_logits[:, -1, :]
        if deterministic:
            formation = fm_logits.argmax(dim=-1)
        else:
            fm_probs = F.softmax(fm_logits / temperature, dim=-1)
            formation = torch.multinomial(fm_probs, 1).squeeze(-1)

        # Target index
        tgt_logits = logits["target_logits"]
        if tgt_logits.dim() == 3:
            tgt_logits = tgt_logits[:, -1, :]
        if deterministic:
            target_index = tgt_logits.argmax(dim=-1)
        else:
            tgt_probs = F.softmax(tgt_logits / temperature, dim=-1)
            target_index = torch.multinomial(tgt_probs, 1).squeeze(-1)

        return {
            "action_type": action_type,
            "target_position": target_pos,
            "priority": priority,
            "formation": formation,
            "target_index": target_index,
        }

    def to_gym_action(self, sampled: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Convert sampled tensors to gymnasium action dict."""
        return {
            "action_type": sampled["action_type"].cpu().item(),
            "target_position": sampled["target_position"].cpu().detach().numpy().flatten(),
            "priority": sampled["priority"].cpu().detach().numpy().flatten(),
            "formation": sampled["formation"].cpu().item(),
            "target_index": sampled["target_index"].cpu().item(),
        }

    def log_prob(
        self,
        logits: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability of taken actions for policy gradient."""
        # Action type log prob
        at_logprobs = F.log_softmax(logits["action_type_logits"], dim=-1)
        at_indices = actions["action_type"].unsqueeze(-1)
        at_lp = at_logprobs.gather(-1, at_indices).squeeze(-1)

        # Formation log prob
        fm_logprobs = F.log_softmax(logits["formation_logits"], dim=-1)
        fm_indices = actions["formation"].unsqueeze(-1)
        fm_lp = fm_logprobs.gather(-1, fm_indices).squeeze(-1)

        # Target log prob
        tgt_logprobs = F.log_softmax(logits["target_logits"], dim=-1)
        tgt_indices = actions["target_index"].unsqueeze(-1)
        tgt_lp = tgt_logprobs.gather(-1, tgt_indices).squeeze(-1)

        return at_lp + fm_lp + tgt_lp
