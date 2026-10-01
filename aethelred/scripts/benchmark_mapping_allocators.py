"""Compare trained or frozen offline mapping candidates on matched held-out missions."""

import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from html import escape
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aethelred.learning.mapping_allocator import (
    BalancedTaskAllocator,
    DurationModel,
    train_duration_model,
)
from aethelred.runtime.mapping_allocation import ContextualMappingAllocator, NearestTaskAllocator
from aethelred.simulation.mapping_mission import SCENARIOS, MappingLayout, MappingMissionSimulation
from aethelred.simulation.workload_allocator import MissionWorkloadAllocator


def boundary_violations(simulation: MappingMissionSimulation) -> int:
    """Check actual movement records against authentication and safety evidence."""
    violations = 0
    for unit in simulation.units:
        events = unit.adapter.journal.read_all()
        authenticated = {e["correlation_id"] for e in events if e["event_type"] == "intent_authenticated"}
        authorised = {e["payload"]["command_id"] for e in events
                      if e["event_type"] == "safety_decision" and e["payload"]["outcome"] == "authorised"}
        starts = {e["correlation_id"]: e["payload"] for e in events
                  if e["event_type"] == "command_execution_started"}
        executed = {e["correlation_id"] for e in events if e["event_type"] == "command_executed"}
        for event in events:
            if event["event_type"] != "mapping_vehicle_moved":
                continue
            command_id = event["correlation_id"]
            payload = starts.get(command_id, {})
            if (command_id not in authorised or command_id not in executed
                    or payload.get("proposal_id") not in authenticated):
                violations += 1
    return violations


def run_case(case: tuple[Path, int, str, str, Path]) -> dict:
    """Independent process job; no shared journals, fitting, or runtime bypass."""
    output, seed, scenario, name, model_path = case
    if name == "nearest-task":
        allocator = NearestTaskAllocator()
    elif name == "analytic-balanced":
        allocator = BalancedTaskAllocator()
    elif name == "learned-balanced":
        allocator = BalancedTaskAllocator(DurationModel.load(model_path))
    elif name == "mission-workload":
        allocator = MissionWorkloadAllocator()
    else:
        raise ValueError("Unknown benchmark allocator")
    proposal_times: list[float] = []
    method = "propose_with_context" if isinstance(allocator, ContextualMappingAllocator) else "propose"
    original = getattr(allocator, method)

    def measured_proposal(*args):
        started = perf_counter()
        try:
            return original(*args)
        finally:
            proposal_times.append(perf_counter() - started)

    setattr(allocator, method, measured_proposal)
    layout = MappingLayout.from_seed(seed)
    simulation = MappingMissionSimulation(output / f"seed-{seed}" / scenario / name,
                                          scenario, layout=layout, allocator=allocator)
    result = simulation.run()
    row = {"seed": seed, "allocator": name, "scenario": scenario,
           "layout": asdict(layout), "result": asdict(result),
           "planning_wall_time": {"calls": len(proposal_times), "total_seconds": sum(proposal_times),
                                  "max_seconds": max(proposal_times, default=0.)},
           "boundary_violations": boundary_violations(simulation)}
    (simulation.output / "measurement.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def paired_analysis(rows: list[dict], seeds: tuple[int, ...],
                    scenarios: tuple[str, ...], *, candidate: str = "learned-balanced",
                    baselines: tuple[str, ...] = ("nearest-task", "analytic-balanced")) -> dict:
    """Resample whole layout clusters, keeping matched scenarios together."""
    indexed = {(r["seed"], r["scenario"], r["allocator"]): r for r in rows}
    names = (*baselines, candidate)
    if not baselines or len(set(names)) != len(names):
        raise ValueError("Candidate and baselines must be distinct")
    expected = {(seed, scenario, name) for seed in seeds for scenario in scenarios for name in names}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Paired analysis requires exactly one row per matched case")
    result = {}
    for baseline in baselines:
        metrics = {}
        for metric in ("ticks", "distance_travelled"):
            deltas = [[indexed[seed, scenario, candidate]["result"][metric]
                       - indexed[seed, scenario, baseline]["result"][metric]
                       for scenario in scenarios] for seed in seeds]
            layout_means = np.asarray([mean(values) for values in deltas])
            rng = np.random.default_rng(20261001)
            samples = rng.choice(layout_means, size=(20000, len(seeds)), replace=True).mean(axis=1)
            interval = np.quantile(samples, (.025, .975)).tolist() if len(seeds) > 1 else None
            metrics[metric] = {
                "mean_paired_difference": float(layout_means.mean()),
                "layout_mean_differences": dict(zip(map(str, seeds), layout_means.tolist())),
                "bootstrap_95_percent_interval": interval,
                "layouts_faster_or_shorter": int((layout_means < 0).sum()),
                "layouts_slower_or_longer": int((layout_means > 0).sum()),
                "layouts_tied": int((layout_means == 0).sum()),
                "by_scenario": {scenario: mean(values[i] for values in deltas)
                                for i, scenario in enumerate(scenarios)},
            }
        result[baseline] = metrics
    return {"independent_layout_count": len(seeds), "resampling_unit": "layout seed",
            "bootstrap_seed": 20261001, "bootstrap_resamples": 20000,
            "candidate": candidate,
            "difference_direction": f"{candidate} minus baseline; negative favors candidate",
            "comparisons": result}


def benchmark(output: Path, seeds: tuple[int, ...], scenarios: tuple[str, ...], *,
              model_path: Path | None = None, expected_sha256: str | None = None,
              excluded_seeds: tuple[int, ...] = (), workers: int = 1,
              include_workload: bool = False) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Held-out seeds and scenarios must be non-empty and unique")
    if any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unknown scenario")
    if include_workload and model_path is None:
        raise ValueError("Workload comparison requires a frozen learned comparator model")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Workers must be between 1 and 8")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output directory must be empty")
    # The training split and hyperparameters are fixed before reading held-out layouts.
    training_seeds, validation_seeds = tuple(range(64)), tuple(range(500, 516))
    frozen_bytes = None
    if model_path is not None:
        frozen_bytes = model_path.read_bytes()
        if expected_sha256 is None or sha256(frozen_bytes).hexdigest() != expected_sha256:
            raise ValueError("Frozen model SHA256 does not match expected fingerprint")
        training = json.loads(frozen_bytes)
        for field in ("training_seeds", "validation_seeds"):
            values = training.get(field)
            if (not isinstance(values, list) or not values
                    or any(type(s) is not int for s in values) or len(set(values)) != len(values)):
                raise ValueError("Frozen model split provenance must contain unique integer seeds")
        training_seeds, validation_seeds = tuple(training["training_seeds"]), tuple(training["validation_seeds"])
        if set(training_seeds) & set(validation_seeds):
            raise ValueError("Frozen training and validation splits overlap")
        DurationModel.load(model_path)
    elif expected_sha256 is not None:
        raise ValueError("Expected fingerprint requires a frozen model path")
    if set(seeds) & (set(training_seeds) | set(validation_seeds)):
        raise ValueError("Held-out seeds overlap training or validation")
    if set(seeds) & set(excluded_seeds):
        raise ValueError("Held-out seeds overlap previously evaluated layouts")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=True).stdout.strip())
    source_paths = [Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "src/aethelred/learning/mapping_allocator.py",
                    Path(__file__).resolve().parents[1] / "src/aethelred/simulation/mapping_mission.py",
                    Path(__file__).resolve().parents[1] / "src/aethelred/simulation/workload_allocator.py"]
    source_paths.extend(sorted((Path(__file__).resolve().parents[1] / "src/aethelred/runtime").glob("*.py")))
    source_hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in source_paths}
    model_path = output / "duration-model.json"
    if frozen_bytes is None:
        training = train_duration_model(model_path, training_seeds, validation_seeds)
    else:
        output.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(frozen_bytes)
    model = DurationModel.load(model_path)
    allocators = [NearestTaskAllocator(), BalancedTaskAllocator(), BalancedTaskAllocator(model)]
    if include_workload:
        allocators.append(MissionWorkloadAllocator())
    candidate_name = "mission-workload" if include_workload else "learned-balanced"
    baselines = ("nearest-task", "analytic-balanced", "learned-balanced") if include_workload else (
        "nearest-task", "analytic-balanced")
    candidate_digest = source_hashes["workload_allocator.py"] if include_workload else model.artifact_sha256
    rows = []
    total = len(seeds) * len(scenarios) * len(allocators)
    plan = {"held_out_seeds": seeds, "scenarios": scenarios, "excluded_seeds": excluded_seeds,
            "candidate": candidate_name, "candidate_sha256": candidate_digest,
            "frozen_comparator_model_sha256": model.artifact_sha256,
            "frozen_model": frozen_bytes is not None,
            "source_revision": revision, "source_file_sha256": source_hashes,
            "workers": workers, "independent_layout_count": len(seeds),
            "analysis": "Paired layout-cluster bootstrap, 20000 resamples, seed 20261001, percentile 95% intervals"}
    (output / "evaluation-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    cases = [(output, seed, scenario, a.allocator_id.split("/")[0], model_path)
             for seed in seeds for scenario in scenarios for a in allocators]

    def collect(results):
        for row in results:
            rows.append(row)
            r = row["result"]
            print(f"[{len(rows)}/{total}] seed={row['seed']} {row['scenario']} {row['allocator']}: "
                  f"complete={r['completed']} ticks={r['ticks']} samples={r['visited_samples']}/{r['total_samples']}",
                  flush=True)

    if workers == 1:
        collect(map(run_case, cases))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            collect(executor.map(run_case, cases))
    summary = {}
    for allocator in allocators:
        name = allocator.allocator_id.split("/")[0]
        entries = [r for r in rows if r["allocator"] == name]
        summary[name] = {
            "mission_count": len(entries),
            "completion_rate": mean(float(r["result"]["completed"]) for r in entries),
            "mean_ticks": mean(r["result"]["ticks"] for r in entries),
            "mean_distance": mean(r["result"]["distance_travelled"] for r in entries),
            "duplicate_visits": sum(r["result"]["duplicate_visits"] for r in entries),
            "boundary_violations": sum(r["boundary_violations"] for r in entries),
            "mean_planning_seconds_per_mission": mean(r.get("planning_wall_time", {}).get("total_seconds", 0.)
                                                     for r in entries),
            "max_proposal_seconds": max(r.get("planning_wall_time", {}).get("max_seconds", 0.) for r in entries),
        }
    candidate = summary[candidate_name]
    clears_comparison = (
        candidate["completion_rate"] == 1 and candidate["duplicate_visits"] == 0
        and candidate["boundary_violations"] == 0
        and all(candidate["mean_ticks"] < summary[name]["mean_ticks"]
                for name in baselines))
    if source_hashes != {p.name: sha256(p.read_bytes()).hexdigest() for p in source_paths}:
        raise ValueError("Benchmark source changed during evaluation; evidence retained but report refused")
    if sha256(model_path.read_bytes()).hexdigest() != model.artifact_sha256:
        raise ValueError("Model changed during evaluation; report refused")
    paired = paired_analysis(rows, seeds, scenarios, candidate=candidate_name, baselines=baselines)
    report = {"schema": "mapping-benchmark/v2", "source_revision": revision,
              "source_working_tree_dirty": dirty,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform()},
              "source_file_sha256": source_hashes,
              "training": training, "candidate": candidate_name, "candidate_sha256": candidate_digest,
              "frozen_comparator_model_sha256": model.artifact_sha256,
              "held_out_seeds": list(seeds), "scenarios": list(scenarios),
              "evaluation_plan": plan, "paired_analysis": paired,
              "summary": summary, "candidate_clears_local_comparison": clears_comparison,
              "deployment_approved": False,
              "interpretation": "Matched synthetic missions, descriptive metrics only; no automatic release promotion.",
              "rows": rows}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output / "report.json").write_text(encoded, encoding="utf-8")
    (output / "report.sha256").write_text(
        sha256((output / "report.json").read_bytes()).hexdigest() + "\n", encoding="utf-8")
    table = "".join(f"<tr><td>{escape(name)}</td><td>{s['mission_count']}</td>"
                    f"<td>{s['completion_rate']:.0%}</td><td>{s['mean_ticks']:.2f}</td>"
                    f"<td>{s['mean_distance']:.2f}</td><td>{s['duplicate_visits']}</td>"
                    f"<td>{s['boundary_violations']}</td><td>{s['max_proposal_seconds']:.4f}</td></tr>"
                    for name, s in summary.items())
    paired_table = "".join(
        f"<tr><td>{escape(name)}</td><td>{metric}</td><td>{s['mean_paired_difference']:.2f}</td>"
        f"<td>{escape(str(s['bootstrap_95_percent_interval']))}</td>"
        f"<td>{s['layouts_faster_or_shorter']}/{s['layouts_slower_or_longer']}/{s['layouts_tied']}</td></tr>"
        for name, metrics in paired["comparisons"].items() for metric, s in metrics.items())
    (output / "report.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Aethelred allocator comparison</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:12px;text-align:left}}</style>
<h1>Held-out mapping allocator comparison</h1><p>Seeds: {escape(str(seeds))}. Scenarios: {escape(', '.join(scenarios))}.</p>
<p>Candidate clears local comparison: {clears_comparison}. Deployment approved: false.</p>
<table><tr><th>Allocator</th><th>Missions</th><th>Completed</th><th>Mean ticks</th><th>Mean distance</th>
<th>Duplicate visits</th><th>Command boundary violations</th><th>Max planning seconds</th></tr>{table}</table>
<h2>Paired differences across {len(seeds)} independent layouts</h2>
<p>{escape(candidate_name)} minus baseline: negative favors the candidate. Whole layouts are resampled, preserving correlated scenarios.
Percentile bootstrap intervals describe this synthetic layout generator, not flight performance.</p>
<table><tr><th>Baseline</th><th>Metric</th><th>Mean difference</th><th>95% interval</th>
<th>Better/worse/tied layouts</th></tr>{paired_table}</table>
<p>The analytic and learned planners use the same batch assignment search. The learned model predicts task duration
from geometry; it was fitted offline on separate seeds. Each matched mission uses the same layout, speeds,
fault schedule, and command safety path. These are descriptive results from an abstract movement model.</p>
<p>The optional mission-workload candidate plans all pending routes across observed idle and busy units,
keeping current tasks locked. It minimizes projected completion time; distance breaks ties between retained
minimum-time routes. Future routes are forecasts, not ownership reservations. It cannot predict future faults.</p>
<p>Planning wall time measures proposal calls only, including concurrent-process contention. It is not
a real-time deadline guarantee; the workload planner is simulation-only.</p>
<p>Training MAE: {training['training_mae_ticks']:.3f} ticks. Validation MAE: {training['validation_mae_ticks']:.3f} ticks.</p>
<p>Full evidence, split provenance, source hashes, and individual results are in report.json;
each mission directory contains a readable replay and verified journals.</p></html>""", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9100, 9101, 9102, 9103])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--model", type=Path, help="Evaluate an existing model without retraining")
    parser.add_argument("--expected-sha256", help="Required fingerprint for --model")
    parser.add_argument("--excluded-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-workload", action="store_true", help="Compare remaining-work candidate with all three baselines")
    args = parser.parse_args()
    report = benchmark(args.output, tuple(args.seeds), tuple(args.scenarios), model_path=args.model,
                       expected_sha256=args.expected_sha256, excluded_seeds=tuple(args.excluded_seeds),
                       workers=args.workers, include_workload=args.include_workload)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output.resolve() / 'report.html'}")


if __name__ == "__main__":
    main()
