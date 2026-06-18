"""
Export trained Aethelred model for deployment on real drone hardware.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --checkpoint checkpoints/best_policy.pt
    python scripts/export_model.py --profile-only --budget 50
    python scripts/export_model.py --format onnx --name aethelred_field_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aethelred.deployment.exporter import ModelExporter
from aethelred.tactical_ai.policy import TacticalPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Aethelred Model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to export")
    parser.add_argument("--output-dir", type=str, default="exported_models")
    parser.add_argument("--name", type=str, default="aethelred_v1")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "torchscript", "onnx", "quantized"])
    parser.add_argument("--budget", type=float, default=50.0,
                        help="Latency budget in ms (50ms = 20Hz control loop)")
    parser.add_argument("--profile-only", action="store_true",
                        help="Only run latency profiling, don't export")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("export")

    logger.info("=" * 60)
    logger.info("  PROJECT AETHELRED - Model Export")
    logger.info("=" * 60)

    # Build or load policy. When loading a checkpoint, rebuild with the stored
    # architecture so the weights map exactly (rather than relying on strict=False
    # to silently drop mismatched tensors).
    if args.checkpoint:
        import torch

        from aethelred.config.settings import DecisionTransformerConfig

        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        cfg = checkpoint.get("config") or {}
        tf_cfg = DecisionTransformerConfig(
            state_embed_dim=cfg.get("state_embed_dim", 256),
            action_embed_dim=cfg.get("action_embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 6),
        )
        policy = TacticalPolicy.build(transformer_config=tf_cfg, device=args.device)
        policy.load_policy_weights(checkpoint["policy_weights"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        policy = TacticalPolicy.build(device=args.device)

    exporter = ModelExporter(output_dir=args.output_dir)

    if args.profile_only:
        profile = exporter.profile_latency(policy, budget_ms=args.budget)
        logger.info(str(profile))
        return

    if args.format == "all":
        results = exporter.export_deployment_package(
            policy, name=args.name, budget_ms=args.budget
        )
    elif args.format == "torchscript":
        result = exporter.export_torchscript(policy, name=args.name)
        result.latency = exporter.profile_latency(policy, budget_ms=args.budget)
        results = {"torchscript": result}
    elif args.format == "onnx":
        result = exporter.export_onnx(policy, name=args.name)
        results = {"onnx": result}
    elif args.format == "quantized":
        result = exporter.export_quantized(policy, name=args.name)
        results = {"quantized": result}

    logger.info("")
    logger.info("  Export Summary:")
    for fmt, result in results.items():
        logger.info(f"    {fmt}: {result.path} ({result.size_mb:.1f} MB)")
        if result.latency:
            logger.info(f"      Latency p95: {result.latency.p95_ms:.2f}ms")
            logger.info(f"      Meets budget: {'YES' if result.latency.meets_budget else 'NO'}")
    logger.info("")


if __name__ == "__main__":
    main()
