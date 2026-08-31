"""Knowledge-distillation utility test (was previously dead / would crash)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aethelred.swarm.policy_distiller import PolicyDistiller
from aethelred.swarm.swarm_unit import LightweightPolicy


def _avg_kl(teacher: nn.Module, student: nn.Module, data, temperature: float) -> float:
    total = 0.0
    with torch.no_grad():
        for x, _ in data:
            t = F.softmax(teacher(x) / temperature, dim=-1)
            s = F.log_softmax(student(x) / temperature, dim=-1)
            total += F.kl_div(s, t, reduction="batchmean").item()
    return total / len(data)


def test_distillation_reduces_divergence():
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
    student = LightweightPolicy(input_dim=16, hidden_dim=32, output_dim=8)
    data = [(torch.randn(4, 16), torch.zeros(4, dtype=torch.long)) for _ in range(8)]

    distiller = PolicyDistiller(temperature=3.0)
    before = _avg_kl(teacher, student, data, distiller.temperature)
    distiller.distill(teacher, student, data, num_epochs=40, lr=1e-2)
    after = _avg_kl(teacher, student, data, distiller.temperature)

    assert after < before, f"distillation did not reduce divergence ({before:.4f} -> {after:.4f})"


def test_compress_delta_prunes_small_changes():
    distiller = PolicyDistiller(compression_threshold=0.1)
    delta = {"w": torch.tensor([0.01, 0.5, -0.2, 0.0])}
    compressed = distiller.compress_delta(delta)
    # Only the two entries above threshold survive
    assert int((compressed["w"] != 0).sum()) == 2
