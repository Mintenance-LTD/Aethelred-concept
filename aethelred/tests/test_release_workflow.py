"""End-to-end test for non-deploying release preparation."""

from __future__ import annotations

from uuid import uuid4

from aethelred.deployment.evaluation import EvaluationScenario, HeldOutEvaluator, ScenarioResult
from aethelred.deployment.promotion import HumanApproval, ModelPromotionGate
from aethelred.deployment.release_ledger import ReleaseLedger
from aethelred.deployment.release_workflow import ReleasePreparationWorkflow
from aethelred.runtime.audit import JsonlAuditJournal


def _runner(model_id: str, scenario: EvaluationScenario) -> ScenarioResult:
    score = 0.9 if model_id == "candidate" else 0.7
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        metrics={"mission_success": score},
        safety_checks={"authorisation": True, "geofence": True},
    )


def test_release_workflow_binds_artifact_evidence_approval_and_ledger(tmp_path) -> None:
    model = tmp_path / "candidate.pt"
    model.write_bytes(b"candidate-model")
    journal = JsonlAuditJournal(tmp_path / "release-audit.jsonl")
    workflow = ReleasePreparationWorkflow(
        HeldOutEvaluator(), ModelPromotionGate(), ReleaseLedger(journal)
    )

    prepared = workflow.prepare(
        candidate_id=uuid4(),
        candidate_model_id="candidate",
        baseline_model_id="baseline",
        model_path=model,
        report_path=tmp_path / "evaluation.json",
        scenarios=tuple(EvaluationScenario(f"scenario-{index}", "held-out") for index in range(20)),
        runner=_runner,
        code_revision="abc123",
        configuration={"schema_version": 1},
        observation_schema="operational-observation/v1",
        runtime_target="torchscript",
        approval=HumanApproval.now("reviewer@example.test", "Held-out evidence reviewed"),
    )

    assert prepared.report_path.is_file()
    assert prepared.manifest.evaluation_report_sha256 == prepared.report.evaluation.report_sha256
    assert prepared.registration.release_id
    assert [event["event_type"] for event in journal.read_all()] == ["release_registered"]
