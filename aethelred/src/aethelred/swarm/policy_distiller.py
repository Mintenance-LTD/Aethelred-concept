"""Knowledge distillation: compress mother's policy for swarm unit deployment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aethelred.swarm.swarm_unit import LightweightPolicy


class PolicyDistiller:
    """Distill a teacher policy into the lightweight per-unit student policy.

    The teacher and student must accept the *same* input tensor (e.g. a local
    feature vector) and emit class logits. This is used to compress a richer
    local-feature policy into the tiny on-unit network — not the full Decision
    Transformer, whose sequence I/O is incompatible with a single-tensor forward.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        compression_threshold: float = 1e-5,
    ) -> None:
        self.temperature = temperature
        self.compression_threshold = compression_threshold

    def distill(
        self,
        teacher_model: nn.Module,
        student: LightweightPolicy,
        training_data: list[tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 10,
        lr: float = 1e-3,
    ) -> LightweightPolicy:
        """Knowledge distillation: train student to mimic teacher's output distribution."""
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        teacher_model.eval()
        student.train()

        for epoch in range(num_epochs):
            for inputs, _ in training_data:
                with torch.no_grad():
                    teacher_out = teacher_model(inputs)
                    if isinstance(teacher_out, dict):
                        teacher_logits = teacher_out.get(
                            "action_type_logits",
                            next(iter(teacher_out.values()))
                        )
                    else:
                        teacher_logits = teacher_out

                student_logits = student(inputs)

                # Resize if dimensions don't match
                min_dim = min(student_logits.shape[-1], teacher_logits.shape[-1])
                s_logits = student_logits[..., :min_dim]
                t_logits = teacher_logits[..., :min_dim]

                teacher_probs = F.softmax(t_logits / self.temperature, dim=-1)
                student_log_probs = F.log_softmax(s_logits / self.temperature, dim=-1)
                loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean"
                ) * (self.temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return student

    def compress_delta(self, delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compress weight delta: prune small changes, quantize."""
        compressed = {}
        for name, diff in delta.items():
            mask = diff.abs() > self.compression_threshold
            pruned = diff * mask.float()
            if pruned.abs().sum() > 0:
                compressed[name] = pruned.half().float()
        return compressed

    def estimate_delta_size(self, delta: dict[str, torch.Tensor]) -> int:
        """Estimate compressed delta size in bytes."""
        total = 0
        for tensor in delta.values():
            nonzero = (tensor.abs() > self.compression_threshold).sum().item()
            total += int(nonzero) * 2
        return total
