"""End-to-end, non-deploying preparation of an approved model release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from aethelred.deployment.evaluation import (
    EvaluationReport,
    EvaluationScenario,
    HeldOutEvaluator,
    ScenarioRunner,
)
from aethelred.deployment.model_manifest import ModelManifest
from aethelred.deployment.promotion import HumanApproval, ModelPromotionGate
from aethelred.deployment.release_ledger import ReleaseLedger, ReleaseRegistration


@dataclass(frozen=True)
class ReleasePreparation:
    """Immutable outputs of candidate evaluation, approval, and registration."""

    report_path: Path
    report: EvaluationReport
    manifest: ModelManifest
    registration: ReleaseRegistration


class ReleasePreparationWorkflow:
    """Compose evaluation, manifest, approval, and registration without activation."""

    def __init__(
        self,
        evaluator: HeldOutEvaluator,
        promotion_gate: ModelPromotionGate,
        ledger: ReleaseLedger,
    ) -> None:
        self._evaluator = evaluator
        self._promotion_gate = promotion_gate
        self._ledger = ledger

    def prepare(
        self,
        candidate_id: UUID,
        candidate_model_id: str,
        baseline_model_id: str,
        model_path: str | Path,
        report_path: str | Path,
        scenarios: tuple[EvaluationScenario, ...],
        runner: ScenarioRunner,
        code_revision: str,
        configuration: dict[str, object],
        observation_schema: str,
        runtime_target: str,
        approval: HumanApproval,
    ) -> ReleasePreparation:
        """Prepare and register an approved release; activation is deliberately absent."""
        report = self._evaluator.evaluate(
            candidate_id, candidate_model_id, baseline_model_id, scenarios, runner
        )
        written_report = report.write(report_path)
        manifest = ModelManifest.create(
            model_path,
            written_report,
            code_revision=code_revision,
            configuration=configuration,
            observation_schema=observation_schema,
            runtime_target=runtime_target,
        )
        approved = self._promotion_gate.approve(manifest, report.evaluation, approval)
        registration = self._ledger.register(approved)
        return ReleasePreparation(written_report, report, manifest, registration)
