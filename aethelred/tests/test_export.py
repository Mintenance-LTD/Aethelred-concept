"""Deployment-export tests: the TorchScript path must trace, save, and reload."""

from __future__ import annotations

import torch

from aethelred.deployment.exporter import ModelExporter, _create_dummy_obs, _InferenceWrapper
from aethelred.tactical_ai.policy import TacticalPolicy


def test_torchscript_export_roundtrip(tmp_path):
    policy = TacticalPolicy.build()
    exporter = ModelExporter(output_dir=str(tmp_path))

    result = exporter.export_torchscript(policy, name="unit_test")
    assert result.path.exists()
    assert result.size_bytes > 0

    # Reload the saved artifact and run it — proves it's a usable deployment file.
    loaded = torch.jit.load(str(result.path))
    dummy = _create_dummy_obs(torch.device("cpu"))
    out = loaded(*dummy)
    assert len(out) == 5  # action_type, target_x, target_y, priority, formation


def test_inference_wrapper_is_torchscript_scriptable():
    """Scripting retains tensor-dependent control flow that tracing could freeze."""
    policy = TacticalPolicy.build()
    wrapper = _InferenceWrapper(policy)
    wrapper.set_to_inference_mode()

    scripted = torch.jit.script(wrapper)
    out = scripted(*_create_dummy_obs(torch.device("cpu")))

    assert len(out) == 5


def test_latency_profile_runs(tmp_path):
    policy = TacticalPolicy.build()
    exporter = ModelExporter(output_dir=str(tmp_path))
    profile = exporter.profile_latency(policy, budget_ms=10_000.0, num_warmup=2, num_samples=5)
    assert profile.num_samples == 5
    assert profile.mean_ms >= 0.0
    assert profile.meets_budget  # generous budget on CPU
