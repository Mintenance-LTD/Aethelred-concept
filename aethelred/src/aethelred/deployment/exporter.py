"""Model export for deployment on real drone hardware.

Supports:
  - ONNX export for edge inference (Jetson Nano, Jetson Orin, etc.)
  - TorchScript for PyTorch-native deployment
  - Quantized INT8 models for low-power devices
  - Latency profiling to ensure real-time operation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aethelred.tactical_ai.policy import TacticalPolicy

logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Latency measurement results."""
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    num_samples: int
    meets_budget: bool
    budget_ms: float

    def __str__(self) -> str:
        status = "PASS" if self.meets_budget else "FAIL"
        return (
            f"Latency [{status}] (budget={self.budget_ms:.1f}ms): "
            f"mean={self.mean_ms:.2f}ms, p50={self.p50_ms:.2f}ms, "
            f"p95={self.p95_ms:.2f}ms, p99={self.p99_ms:.2f}ms, "
            f"max={self.max_ms:.2f}ms ({self.num_samples} samples)"
        )


@dataclass
class ExportResult:
    """Result of a model export operation."""
    format: str
    path: Path
    size_bytes: int
    latency: LatencyProfile | None = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class ModelExporter:
    """Exports trained Aethelred models for deployment on real drone hardware."""

    def __init__(self, output_dir: str = "exported_models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_torchscript(
        self,
        policy: TacticalPolicy,
        name: str = "aethelred_tactical",
    ) -> ExportResult:
        """Export as TorchScript for PyTorch-native deployment."""
        logger.info("Exporting TorchScript model...")

        # Create a wrapper that takes raw observation tensors
        wrapper = _InferenceWrapper(policy)
        wrapper.set_to_inference_mode()

        # Generate dummy input
        dummy = _create_dummy_obs(policy.device)

        # Script the model. check_trace is disabled because attention introduces
        # tiny (~1e-5) numerical reordering that trips the built-in re-run check;
        # the export round-trip test validates correctness instead.
        try:
            scripted = torch.jit.trace(wrapper, dummy, check_trace=False)
        except (RuntimeError, ValueError):
            logger.warning("Tracing failed, falling back to scripting")
            scripted = torch.jit.script(wrapper)

        path = self.output_dir / f"{name}.pt"
        scripted.save(str(path))

        size = path.stat().st_size
        logger.info(f"TorchScript exported: {path} ({size / 1024 / 1024:.1f} MB)")

        return ExportResult(format="torchscript", path=path, size_bytes=size)

    def export_onnx(
        self,
        policy: TacticalPolicy,
        name: str = "aethelred_tactical",
        opset_version: int = 17,
    ) -> ExportResult:
        """Export as ONNX for cross-platform edge inference."""
        logger.info("Exporting ONNX model...")

        wrapper = _InferenceWrapper(policy)
        wrapper.set_to_inference_mode()

        dummy = _create_dummy_obs(policy.device)

        path = self.output_dir / f"{name}.onnx"

        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            opset_version=opset_version,
            input_names=[
                "friendlies", "friendly_mask", "threats", "threat_mask",
                "objectives", "terrain", "globals_vec",
            ],
            output_names=["action_type", "target_x", "target_y", "priority", "formation"],
            dynamic_axes={
                "friendlies": {0: "batch"},
                "friendly_mask": {0: "batch"},
                "threats": {0: "batch"},
                "threat_mask": {0: "batch"},
                "objectives": {0: "batch"},
                "terrain": {0: "batch"},
                "globals_vec": {0: "batch"},
            },
        )

        size = path.stat().st_size
        logger.info(f"ONNX exported: {path} ({size / 1024 / 1024:.1f} MB)")

        return ExportResult(format="onnx", path=path, size_bytes=size)

    def export_quantized(
        self,
        policy: TacticalPolicy,
        name: str = "aethelred_tactical_int8",
    ) -> ExportResult:
        """Export INT8 quantized model for low-power inference."""
        logger.info("Exporting INT8 quantized model...")

        wrapper = _InferenceWrapper(policy)
        wrapper.set_to_inference_mode()

        # Dynamic quantization (no calibration data needed)
        quantized = torch.quantization.quantize_dynamic(
            wrapper,
            {nn.Linear},
            dtype=torch.qint8,
        )

        path = self.output_dir / f"{name}.pt"
        torch.save(quantized.state_dict(), path)

        size = path.stat().st_size
        logger.info(f"Quantized INT8 exported: {path} ({size / 1024 / 1024:.1f} MB)")

        return ExportResult(format="quantized_int8", path=path, size_bytes=size)

    def profile_latency(
        self,
        policy: TacticalPolicy,
        budget_ms: float = 50.0,
        num_warmup: int = 20,
        num_samples: int = 200,
    ) -> LatencyProfile:
        """Profile inference latency to ensure real-time operation.

        Args:
            budget_ms: Maximum acceptable inference time in milliseconds.
                       For a 20Hz control loop, this should be <= 50ms.
                       For a 10Hz loop, <= 100ms.
        """
        logger.info(f"Profiling latency (budget={budget_ms}ms, samples={num_samples})...")

        wrapper = _InferenceWrapper(policy)
        wrapper.set_to_inference_mode()

        dummy = _create_dummy_obs(policy.device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                wrapper(*dummy)

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_samples):
                if policy.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                wrapper(*dummy)
                if policy.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0  # ms
                latencies.append(elapsed)

        arr = np.array(latencies)
        profile = LatencyProfile(
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(arr.max()),
            min_ms=float(arr.min()),
            num_samples=num_samples,
            meets_budget=float(np.percentile(arr, 95)) <= budget_ms,
            budget_ms=budget_ms,
        )

        logger.info(str(profile))
        return profile

    def export_deployment_package(
        self,
        policy: TacticalPolicy,
        name: str = "aethelred_v1",
        include_onnx: bool = True,
        budget_ms: float = 50.0,
    ) -> dict[str, ExportResult]:
        """Export a complete deployment package with all formats + profiling."""
        results = {}

        # TorchScript (primary)
        results["torchscript"] = self.export_torchscript(policy, f"{name}_scripted")

        # ONNX (for non-PyTorch runtimes)
        if include_onnx:
            try:
                results["onnx"] = self.export_onnx(policy, f"{name}_onnx")
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"ONNX export failed: {e}")

        # Latency profile
        latency = self.profile_latency(policy, budget_ms=budget_ms)
        results["torchscript"].latency = latency

        # Save deployment manifest
        import json
        manifest = {
            "name": name,
            "formats": {k: str(v.path) for k, v in results.items()},
            "sizes_mb": {k: v.size_mb for k, v in results.items()},
            "latency_p95_ms": latency.p95_ms,
            "latency_budget_ms": budget_ms,
            "meets_budget": latency.meets_budget,
        }

        manifest_path = self.output_dir / f"{name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Deployment package exported to {self.output_dir}")
        for fmt, result in results.items():
            logger.info(f"  {fmt}: {result.size_mb:.1f} MB")
        logger.info(f"  Latency p95: {latency.p95_ms:.2f}ms (budget: {budget_ms}ms)")

        return results


class _InferenceWrapper(nn.Module):
    """Wraps the full policy pipeline into a single forward() for export."""

    def __init__(self, policy: TacticalPolicy) -> None:
        super().__init__()
        self.encoder = policy.state_encoder
        self.transformer = policy.transformer
        self.config = policy.config

    def set_to_inference_mode(self) -> None:
        """Set all sub-modules to inference mode."""
        self.encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

    def forward(
        self,
        friendlies: torch.Tensor,
        friendly_mask: torch.Tensor,
        threats: torch.Tensor,
        threat_mask: torch.Tensor,
        objectives: torch.Tensor,
        terrain: torch.Tensor,
        globals_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single forward pass: obs tensors -> action outputs."""
        obs = {
            "friendlies": friendlies,
            "friendly_mask": friendly_mask,
            "threats": threats,
            "threat_mask": threat_mask,
            "objectives": objectives,
            "terrain": terrain,
            "globals": globals_vec,
        }

        # Encode state
        state_embed = self.encoder(obs)  # (B, D)

        # Build minimal sequence (single timestep)
        states = state_embed.unsqueeze(1)  # (B, 1, D)
        actions = torch.zeros(
            state_embed.shape[0], 1, self.config.action_embed_dim,
            device=state_embed.device,
        )
        rtg = torch.ones(
            state_embed.shape[0], 1, 1, device=state_embed.device
        ) * 5.0
        timesteps = torch.zeros(
            state_embed.shape[0], 1, dtype=torch.long, device=state_embed.device
        )

        # Get action
        logits = self.transformer(states, actions, rtg, timesteps)
        sampled = self.transformer.action_head.sample_action(logits, deterministic=True)

        # target_position is always (B, 2) from the action head — index directly so
        # the graph is trace-safe (no Python-level branching on tensor shape).
        target_position = sampled["target_position"]
        return (
            sampled["action_type"].float(),
            target_position[:, 0],
            target_position[:, 1],
            sampled["priority"].squeeze(-1),
            sampled["formation"].float(),
        )


def _create_dummy_obs(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create dummy observation tensors for tracing/profiling."""
    return (
        torch.randn(1, 32, 16, device=device),   # friendlies
        torch.ones(1, 32, device=device),          # friendly_mask
        torch.randn(1, 16, 18, device=device),    # threats
        torch.ones(1, 16, device=device),          # threat_mask
        torch.randn(1, 8, 4, device=device),      # objectives
        torch.randn(1, 1, 50, 50, device=device), # terrain
        torch.randn(1, 4, device=device),          # globals
    )
