"""Evaluation and human-approval gates for model release candidates.

This module creates governance records only. It deliberately contains no model
loading, weight copying, command dispatch, or deployment operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from uuid import UUID

from aethelred.deployment.model_manifest import ModelManifest


class PromotionError(ValueError):
    """Raised when a candidate lacks the evidence needed for promotion."""


@dataclass(frozen=True)
class HeldOutEvaluation:
    """Comparable candidate and baseline measurements from held-out scenarios."""

    candidate_id: UUID
    scenario_count: int
    candidate_metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    safety_checks: Mapping[str, bool]
    report_sha256: str

    def __post_init__(self) -> None:
        if self.scenario_count <= 0:
            raise PromotionError("Held-out evaluation must include at least one scenario")
        if len(self.report_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.report_sha256):
            raise PromotionError("Evaluation report hash must be a lowercase SHA-256 digest")
        for name, value in {**self.candidate_metrics, **self.baseline_metrics}.items():
            if not isfinite(value):
                raise PromotionError(f"Metric {name!r} must be finite")


@dataclass(frozen=True)
class PromotionPolicy:
    """Minimum evidence requirements, configurable per approved programme."""

    minimum_scenarios: int = 20
    required_metrics: tuple[str, ...] = ("mission_success",)
    require_strict_improvement: bool = True


@dataclass(frozen=True)
class HumanApproval:
    """A named, accountable approval for an already-evaluated candidate."""

    approver: str
    rationale: str
    approved_at: datetime

    def __post_init__(self) -> None:
        if not self.approver.strip() or not self.rationale.strip():
            raise PromotionError("Approver identity and rationale are required")
        if self.approved_at.tzinfo is None:
            raise PromotionError("Approval time must be timezone-aware")

    @classmethod
    def now(cls, approver: str, rationale: str) -> HumanApproval:
        """Create a human approval record with an auditable UTC timestamp."""
        return cls(approver=approver, rationale=rationale, approved_at=datetime.now(UTC))


@dataclass(frozen=True)
class ApprovedModelRelease:
    """An approved record, intentionally separate from deployment or actuation."""

    manifest: ModelManifest
    evaluation: HeldOutEvaluation
    approval: HumanApproval


class ModelPromotionGate:
    """Validate evaluation evidence before a human can approve a release record."""

    def __init__(self, policy: PromotionPolicy | None = None) -> None:
        self.policy = policy or PromotionPolicy()

    def assess(self, evaluation: HeldOutEvaluation) -> tuple[str, ...]:
        """Return all reasons an evaluation is ineligible; empty means eligible."""
        reasons: list[str] = []
        if evaluation.scenario_count < self.policy.minimum_scenarios:
            reasons.append("insufficient held-out scenarios")
        failed_checks = sorted(name for name, passed in evaluation.safety_checks.items() if not passed)
        if failed_checks:
            reasons.append(f"failed safety checks: {', '.join(failed_checks)}")
        if not evaluation.safety_checks:
            reasons.append("no safety checks recorded")
        for metric in self.policy.required_metrics:
            if metric not in evaluation.candidate_metrics or metric not in evaluation.baseline_metrics:
                reasons.append(f"missing required metric: {metric}")
                continue
            candidate = evaluation.candidate_metrics[metric]
            baseline = evaluation.baseline_metrics[metric]
            improved = candidate > baseline if self.policy.require_strict_improvement else candidate >= baseline
            if not improved:
                reasons.append(f"candidate did not improve required metric: {metric}")
        return tuple(reasons)

    def approve(
        self,
        manifest: ModelManifest,
        evaluation: HeldOutEvaluation,
        approval: HumanApproval,
    ) -> ApprovedModelRelease:
        """Create an approval record after validating immutable release evidence."""
        reasons = self.assess(evaluation)
        if reasons:
            raise PromotionError("Candidate is not eligible: " + "; ".join(reasons))
        if manifest.evaluation_report_sha256 != evaluation.report_sha256:
            raise PromotionError("Manifest evaluation hash does not match supplied evidence")
        return ApprovedModelRelease(manifest=manifest, evaluation=evaluation, approval=approval)
