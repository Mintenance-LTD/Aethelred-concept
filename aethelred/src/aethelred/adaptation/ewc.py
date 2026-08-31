"""Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from aethelred.config.settings import EWCConfig


class EWCRegularizer:
    """
    EWC prevents catastrophic forgetting by penalizing changes to
    weights that are important for previously learned tasks.

    After mastering a threat type, compute the Fisher Information Matrix
    to identify critical weights. Future adaptations are regularized
    to preserve those weights.
    """

    def __init__(self, config: EWCConfig) -> None:
        self.lambda_ewc = config.lambda_ewc
        self.fisher_sample_size = config.fisher_sample_size
        self.fisher_matrices: dict[str, dict[str, torch.Tensor]] = {}
        self.optimal_params: dict[str, dict[str, torch.Tensor]] = {}

    def register_task(
        self,
        task_name: str,
        model: nn.Module,
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: callable,
    ) -> None:
        """
        After model masters a task, compute and store Fisher Information.

        Args:
            task_name: identifier (e.g. "counter_small_arms")
            model: the trained model
            data_iterator: iterator yielding (input, target) pairs
            loss_fn: the loss function used during training
        """
        fisher = self._compute_fisher(model, data_iterator, loss_fn)
        self.fisher_matrices[task_name] = fisher
        self.optimal_params[task_name] = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term.
        Add this to the training loss to prevent forgetting.

        penalty = lambda * sum_tasks( sum_params( F_i * (theta - theta_i*)^2 ) )
        """
        device = next(model.parameters()).device
        total_penalty = torch.tensor(0.0, device=device)

        for task_name in self.fisher_matrices:
            fisher = self.fisher_matrices[task_name]
            optimal = self.optimal_params[task_name]

            for name, param in model.named_parameters():
                if name in fisher and name in optimal:
                    f = fisher[name].to(device)
                    o = optimal[name].to(device)
                    total_penalty = total_penalty + (f * (param - o) ** 2).sum()

        return self.lambda_ewc * total_penalty

    @property
    def num_tasks(self) -> int:
        return len(self.fisher_matrices)

    @property
    def registered_tasks(self) -> list[str]:
        return list(self.fisher_matrices.keys())

    def _compute_fisher(
        self,
        model: nn.Module,
        data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: callable,
    ) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix via empirical estimate."""
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        model.train()
        count = 0

        for inputs, targets in data_iterator:
            if count >= self.fisher_sample_size:
                break

            model.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, dict):
                # Handle Decision Transformer dict output
                pred = outputs.get(
                    "action_type_logits",
                    next(iter(outputs.values()))
                )
                loss = loss_fn(pred, targets)
            else:
                loss = loss_fn(outputs, targets)

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] = fisher[name] + param.grad.data ** 2

            count += 1

        # Average
        for name, value in fisher.items():
            fisher[name] = value / max(count, 1)

        return fisher
