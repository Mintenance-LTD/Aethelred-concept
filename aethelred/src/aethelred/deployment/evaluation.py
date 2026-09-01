"""Deterministic held-out scenario evaluation for model-promotion evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from aethelred.deployment.promotion import HeldOutEvaluation, PromotionError


class OperationalScenarioCategory(StrEnum):
    """Non-offensive contexts that a releasable policy may be evaluated against."""

    SURVEY = "survey"
    INSPECTION = "inspection"
    MAPPING = "mapping"
    RELAY = "relay"
    DISASTER_SEARCH = "disaster_search"
    DEGRADED_GPS = "degraded_gps"
    DEGRADED_COMMS = "degraded_comms"
    ADVERSE_WEATHER = "adverse_weather"


@dataclass(frozen=True)
class EvaluationScenario:
    """A declared held-out scenario; no training data belongs in this contract."""

    scenario_id: str
    distribution: str
    category: OperationalScenarioCategory = OperationalScenarioCategory.SURVEY

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.distribution.strip():
            raise PromotionError("Held-out scenarios require an ID and distribution label")


@dataclass(frozen=True)
class ScenarioResult:
    """One candidate or baseline result from the same held-out scenario."""

    scenario_id: str
    metrics: Mapping[str, float]
    safety_checks: Mapping[str, bool]


ScenarioRunner = Callable[[str, EvaluationScenario], ScenarioResult]


@dataclass(frozen=True)
class EvaluationReport:
    """Canonical report that binds scenario-level evidence to its SHA-256 digest."""

    evaluation: HeldOutEvaluation
    candidate_model_id: str
    baseline_model_id: str
    scenarios: tuple[EvaluationScenario, ...]
    candidate_results: tuple[ScenarioResult, ...]
    baseline_results: tuple[ScenarioResult, ...]

    def to_json(self) -> str:
        """Return stable JSON used both for review and hash-bound promotion."""
        return json.dumps(self._evidence_payload(), sort_keys=True, separators=(",", ":")) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write the immutable evaluation report without altering its evidence."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
        return destination

    def _evidence_payload(self) -> dict[str, object]:
        """Return the exact payload covered by ``evaluation.report_sha256``."""
        return {
            "candidate_id": str(self.evaluation.candidate_id),
            "candidate_model_id": self.candidate_model_id,
            "baseline_model_id": self.baseline_model_id,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "candidate_results": [asdict(result) for result in self.candidate_results],
            "baseline_results": [asdict(result) for result in self.baseline_results],
            "candidate_metrics": dict(self.evaluation.candidate_metrics),
            "baseline_metrics": dict(self.evaluation.baseline_metrics),
            "safety_checks": dict(self.evaluation.safety_checks),
            "scenario_categories": self.evaluation.scenario_categories,
        }


class HeldOutEvaluator:
    """Run identical declared scenarios for a candidate and its baseline."""

    def evaluate(
        self,
        candidate_id: UUID,
        candidate_model_id: str,
        baseline_model_id: str,
        scenarios: tuple[EvaluationScenario, ...],
        runner: ScenarioRunner,
    ) -> EvaluationReport:
        """Produce hash-bound aggregate evidence from pairwise held-out runs."""
        self._validate_inputs(candidate_model_id, baseline_model_id, scenarios)
        candidate_results = tuple(runner(candidate_model_id, scenario) for scenario in scenarios)
        baseline_results = tuple(runner(baseline_model_id, scenario) for scenario in scenarios)
        self._validate_results(scenarios, candidate_results, baseline_results)
        candidate_metrics = self._mean_metrics(candidate_results)
        baseline_metrics = self._mean_metrics(baseline_results)
        safety_checks = self._aggregate_safety(candidate_results)
        scenario_categories = tuple(sorted({scenario.category.value for scenario in scenarios}))
        evidence_payload = {
            "candidate_id": str(candidate_id),
            "candidate_model_id": candidate_model_id,
            "baseline_model_id": baseline_model_id,
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "candidate_results": [asdict(result) for result in candidate_results],
            "baseline_results": [asdict(result) for result in baseline_results],
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "safety_checks": safety_checks,
            "scenario_categories": scenario_categories,
        }
        report_hash = hashlib.sha256(
            (json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest()
        evaluation = HeldOutEvaluation(
            candidate_id=candidate_id,
            scenario_count=len(scenarios),
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            safety_checks=safety_checks,
            report_sha256=report_hash,
            scenario_categories=scenario_categories,
        )
        return EvaluationReport(
            evaluation=evaluation,
            candidate_model_id=candidate_model_id,
            baseline_model_id=baseline_model_id,
            scenarios=scenarios,
            candidate_results=candidate_results,
            baseline_results=baseline_results,
        )

    @staticmethod
    def _validate_inputs(
        candidate_model_id: str,
        baseline_model_id: str,
        scenarios: tuple[EvaluationScenario, ...],
    ) -> None:
        if not candidate_model_id.strip() or not baseline_model_id.strip():
            raise PromotionError("Candidate and baseline model IDs are required")
        if candidate_model_id == baseline_model_id:
            raise PromotionError("Candidate and baseline model IDs must differ")
        if not scenarios:
            raise PromotionError("At least one held-out scenario is required")
        ids = [scenario.scenario_id for scenario in scenarios]
        if len(ids) != len(set(ids)):
            raise PromotionError("Held-out scenario IDs must be unique")

    @staticmethod
    def _validate_results(
        scenarios: tuple[EvaluationScenario, ...],
        candidate_results: tuple[ScenarioResult, ...],
        baseline_results: tuple[ScenarioResult, ...],
    ) -> None:
        expected_ids = tuple(scenario.scenario_id for scenario in scenarios)
        for results, label in ((candidate_results, "candidate"), (baseline_results, "baseline")):
            if tuple(result.scenario_id for result in results) != expected_ids:
                raise PromotionError(f"{label} results do not match the held-out scenario set")
            for result in results:
                if not result.metrics:
                    raise PromotionError(f"{label} result {result.scenario_id!r} has no metrics")
                if any(not math.isfinite(value) for value in result.metrics.values()):
                    raise PromotionError(f"{label} result {result.scenario_id!r} has a non-finite metric")
        candidate_metric_names = set(candidate_results[0].metrics)
        baseline_metric_names = set(baseline_results[0].metrics)
        if candidate_metric_names != baseline_metric_names:
            raise PromotionError("Candidate and baseline metric names must match")
        for result in (*candidate_results, *baseline_results):
            if set(result.metrics) != candidate_metric_names:
                raise PromotionError("Every scenario result must provide the same metric names")

    @staticmethod
    def _mean_metrics(results: tuple[ScenarioResult, ...]) -> dict[str, float]:
        return {
            metric: sum(result.metrics[metric] for result in results) / len(results)
            for metric in sorted(results[0].metrics)
        }

    @staticmethod
    def _aggregate_safety(results: tuple[ScenarioResult, ...]) -> dict[str, bool]:
        names = set().union(*(result.safety_checks.keys() for result in results))
        return {name: all(result.safety_checks.get(name, False) for result in results) for name in sorted(names)}
