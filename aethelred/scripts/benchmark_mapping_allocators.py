"""Train an offline mapping candidate and compare matched held-out missions."""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from hashlib import sha256
from html import escape
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aethelred.learning.mapping_allocator import (
    BalancedTaskAllocator,
    DurationModel,
    train_duration_model,
)
from aethelred.runtime.mapping_allocation import NearestTaskAllocator
from aethelred.simulation.mapping_mission import SCENARIOS, MappingLayout, MappingMissionSimulation


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


def benchmark(output: Path, seeds: tuple[int, ...], scenarios: tuple[str, ...]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Held-out seeds and scenarios must be non-empty and unique")
    if any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Unknown scenario")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output directory must be empty")
    # The training split and hyperparameters are fixed before reading held-out layouts.
    training_seeds, validation_seeds = tuple(range(64)), tuple(range(500, 516))
    if set(seeds) & (set(training_seeds) | set(validation_seeds)):
        raise ValueError("Held-out seeds overlap training or validation")
    model_path = output / "duration-model.json"
    training = train_duration_model(model_path, training_seeds, validation_seeds)
    model = DurationModel.load(model_path)
    allocators = (NearestTaskAllocator(), BalancedTaskAllocator(), BalancedTaskAllocator(model))
    rows = []
    total = len(seeds) * len(scenarios) * len(allocators)
    for seed in seeds:
        layout = MappingLayout.from_seed(seed)
        for scenario in scenarios:
            for allocator in allocators:
                name = allocator.allocator_id.split("/")[0]
                simulation = MappingMissionSimulation(output / f"seed-{seed}" / scenario / name,
                                                      scenario, layout=layout, allocator=allocator)
                result = simulation.run()
                rows.append({"seed": seed, "allocator": name, "scenario": scenario,
                             "layout": asdict(layout), "result": asdict(result),
                             "boundary_violations": boundary_violations(simulation)})
                print(f"[{len(rows)}/{total}] seed={seed} {scenario} {name}: "
                      f"complete={result.completed} ticks={result.ticks} samples={result.visited_samples}/{result.total_samples}",
                      flush=True)
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
        }
    candidate = summary["learned-balanced"]
    clears_comparison = (
        candidate["completion_rate"] == 1 and candidate["duplicate_visits"] == 0
        and candidate["boundary_violations"] == 0
        and all(candidate["mean_ticks"] < summary[name]["mean_ticks"]
                for name in ("nearest-task", "analytic-balanced")))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=True).stdout.strip())
    source_paths = [Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "src/aethelred/learning/mapping_allocator.py",
                    Path(__file__).resolve().parents[1] / "src/aethelred/simulation/mapping_mission.py"]
    source_paths.extend(sorted((Path(__file__).resolve().parents[1] / "src/aethelred/runtime").glob("*.py")))
    report = {"schema": "mapping-benchmark/v1", "source_revision": revision,
              "source_working_tree_dirty": dirty,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform()},
              "source_file_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "training": training, "candidate_sha256": model.artifact_sha256,
              "held_out_seeds": list(seeds), "scenarios": list(scenarios),
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
                    f"<td>{s['boundary_violations']}</td></tr>" for name, s in summary.items())
    (output / "report.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Aethelred allocator comparison</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:12px;text-align:left}}</style>
<h1>Held-out mapping allocator comparison</h1><p>Seeds: {escape(str(seeds))}. Scenarios: {escape(', '.join(scenarios))}.</p>
<p>Candidate clears local comparison: {clears_comparison}. Deployment approved: false.</p>
<table><tr><th>Allocator</th><th>Missions</th><th>Completed</th><th>Mean ticks</th><th>Mean distance</th>
<th>Duplicate visits</th><th>Command boundary violations</th></tr>{table}</table>
<p>The analytic and learned planners use the same batch assignment search. The learned model predicts task duration
from geometry; it was fitted offline on separate seeds. Each matched mission uses the same layout, speeds,
fault schedule, and command safety path. These are descriptive results from an abstract movement model.</p>
<p>Training MAE: {training['training_mae_ticks']:.3f} ticks. Validation MAE: {training['validation_mae_ticks']:.3f} ticks.</p>
<p>Full evidence, split provenance, source hashes, and individual results are in report.json;
each mission directory contains a readable replay and verified journals.</p></html>""", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9100, 9101, 9102, 9103])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    report = benchmark(args.output, tuple(args.seeds), tuple(args.scenarios))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output.resolve() / 'report.html'}")


if __name__ == "__main__":
    main()
