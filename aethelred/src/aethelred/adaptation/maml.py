"""Model-Agnostic Meta-Learning (MAML) for few-shot tactical adaptation."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn

from aethelred.config.settings import MAMLConfig


class MAMLAdapter:
    """
    MAML for few-shot adaptation to new threat types.
    When a new threat is encountered, MAML enables rapid adaptation
    from very few examples (inner loop gradient steps).
    """

    def __init__(self, config: MAMLConfig) -> None:
        self.inner_lr = config.inner_lr
        self.inner_steps = config.inner_steps
        self.meta_lr = config.meta_lr

    def inner_loop_adapt(
        self,
        model: nn.Module,
        support_data: list[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> nn.Module:
        """
        Few-shot adaptation via gradient descent on a support set.
        Returns an adapted copy of the model without modifying the original.

        Args:
            model: the base model to adapt
            support_data: list of (input, target) pairs
            loss_fn: loss function(prediction, target) -> scalar loss
        """
        adapted_model = copy.deepcopy(model)
        adapted_model.train()

        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        for step in range(self.inner_steps):
            total_loss = torch.tensor(0.0, device=next(adapted_model.parameters()).device)

            for inputs, targets in support_data:
                predictions = adapted_model(inputs)
                if isinstance(predictions, dict):
                    # Decision transformer returns dict of logits
                    pred_tensor = predictions.get(
                        "action_type_logits",
                        next(iter(predictions.values()))
                    )
                    loss = loss_fn(pred_tensor, targets)
                else:
                    loss = loss_fn(predictions, targets)
                total_loss = total_loss + loss

            total_loss = total_loss / max(len(support_data), 1)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            optimizer.step()

        return adapted_model

    def meta_train_step(
        self,
        model: nn.Module,
        meta_optimizer: torch.optim.Optimizer,
        task_batch: list[dict[str, list[tuple[torch.Tensor, torch.Tensor]]]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        """
        One step of MAML meta-training.

        Args:
            model: the meta-model to train
            meta_optimizer: optimizer for outer loop
            task_batch: list of tasks, each with 'support' and 'query' data
            loss_fn: loss function

        Returns:
            Average meta-loss across tasks
        """
        meta_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for task in task_batch:
            # Inner loop: adapt on support set
            adapted_model = self.inner_loop_adapt(
                model, task["support"], loss_fn
            )

            # Outer loop: evaluate on query set
            for inputs, targets in task["query"]:
                predictions = adapted_model(inputs)
                if isinstance(predictions, dict):
                    pred_tensor = predictions.get(
                        "action_type_logits",
                        next(iter(predictions.values()))
                    )
                    loss = loss_fn(pred_tensor, targets)
                else:
                    loss = loss_fn(predictions, targets)
                meta_loss = meta_loss + loss

        meta_loss = meta_loss / max(len(task_batch), 1)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_optimizer.step()

        return meta_loss.item()
