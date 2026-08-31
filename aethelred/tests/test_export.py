"""Deployment-export tests: the TorchScript path must trace, save, and reload."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
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


def test_eager_torchscript_and_onnx_exports_have_behavioural_parity(tmp_path):
    policy = TacticalPolicy.build()
    exporter = ModelExporter(output_dir=str(tmp_path))
    wrapper = _InferenceWrapper(policy)
    wrapper.set_to_inference_mode()
    dummy = _create_dummy_obs(torch.device("cpu"))

    with torch.no_grad():
        eager_outputs = wrapper(*dummy)
    torchscript = exporter.export_torchscript(policy, name="parity")
    scripted_outputs = torch.jit.load(str(torchscript.path))(*dummy)
    onnx_result = exporter.export_onnx(policy, name="parity")
    onnx.checker.check_model(str(onnx_result.path))
    session = ort.InferenceSession(str(onnx_result.path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(
        None,
        {
            name: value.detach().cpu().numpy()
            for name, value in zip((item.name for item in session.get_inputs()), dummy, strict=True)
        },
    )

    for eager, scripted, exported in zip(eager_outputs, scripted_outputs, onnx_outputs, strict=True):
        np.testing.assert_allclose(scripted.detach().cpu().numpy(), eager.detach().cpu().numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(exported, eager.detach().cpu().numpy(), rtol=1e-4, atol=1e-5)


def test_latency_profile_runs(tmp_path):
    policy = TacticalPolicy.build()
    exporter = ModelExporter(output_dir=str(tmp_path))
    profile = exporter.profile_latency(policy, budget_ms=10_000.0, num_warmup=2, num_samples=5)
    assert profile.num_samples == 5
    assert profile.mean_ms >= 0.0
    assert profile.meets_budget  # generous budget on CPU
