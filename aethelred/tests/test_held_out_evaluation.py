"""Tests for deterministic held-out promotion evidence."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from aethelred.deployment.evaluation import (
    EvaluationScenario,
    HeldOutEvaluator,
    OperationalScenarioCategory,
    ScenarioResult,
)
from aethelred.deployment.promotion import PromotionError


def _runner(model_id: str, scenario: EvaluationScenario) -> ScenarioResult:
    candidate = model_id == "candidate-v2"
    base = 0.8 if candidate else 0.6
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        metrics={"mission_success": base + (0.1 if scenario.scenario_id == "weather" else 0.0)},
        safety_checks={"authorisation": True, "geofence": True},
    )


def test_held_out_evaluator_generates_reproducible_hash_bound_evidence(tmp_path) -> None:
    scenarios = (
        EvaluationScenario("nominal", "nominal"),
        EvaluationScenario("weather", "adverse-weather"),
    )
    report = HeldOutEvaluator().evaluate(uuid4(), "candidate-v2", "baseline-v1", scenarios, _runner)
    destination = report.write(tmp_path / "evaluation.json")

    assert report.evaluation.candidate_metrics["mission_success"] == pytest.approx(0.85)
    assert report.evaluation.baseline_metrics["mission_success"] == pytest.approx(0.65)
    assert report.evaluation.safety_checks == {"authorisation": True, "geofence": True}
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == report.evaluation.report_sha256
    assert len(report.evaluation.report_sha256) == 64


def test_evaluator_rejects_invalid_or_mismatched_held_out_inputs() -> None:
    evaluator = HeldOutEvaluator()
    scenario = EvaluationScenario("nominal", "nominal")
    with pytest.raises(PromotionError, match="must differ"):
        evaluator.evaluate(uuid4(), "same", "same", (scenario,), _runner)
    with pytest.raises(PromotionError, match="unique"):
        evaluator.evaluate(uuid4(), "candidate-v2", "baseline-v1", (scenario, scenario), _runner)


def test_evaluator_records_non_offensive_scenario_categories() -> None:
    scenarios = (
        EvaluationScenario("survey", "held-out", OperationalScenarioCategory.SURVEY),
        EvaluationScenario("gps-loss", "held-out", OperationalScenarioCategory.DEGRADED_GPS),
    )

    report = HeldOutEvaluator().evaluate(uuid4(), "candidate-v2", "baseline-v1", scenarios, _runner)

    assert report.evaluation.scenario_categories == ("degraded_gps", "survey")
