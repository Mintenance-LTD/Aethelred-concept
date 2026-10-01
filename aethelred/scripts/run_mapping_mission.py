"""Run a civilian mapping mission and save its complete audit replay."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aethelred.learning.mapping_allocator import BalancedTaskAllocator, DurationModel
from aethelred.runtime.mapping_allocation import NearestTaskAllocator
from aethelred.simulation.mapping_mission import SCENARIOS, MappingLayout, MappingMissionSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="unit_loss")
    parser.add_argument("--output", type=Path, required=True, help="Empty directory for mission evidence")
    parser.add_argument("--max-ticks", type=int, default=180)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--allocator", choices=("nearest", "analytic", "learned"), default="nearest")
    parser.add_argument("--model", type=Path, help="Offline duration model, required for the learned allocator")
    args = parser.parse_args()
    if args.allocator == "learned" and args.model is None:
        parser.error("--allocator learned requires --model")
    if args.allocator != "learned" and args.model is not None:
        parser.error("--model is only used by --allocator learned")
    allocator = (BalancedTaskAllocator(DurationModel.load(args.model)) if args.model is not None
                 else BalancedTaskAllocator() if args.allocator == "analytic" else NearestTaskAllocator())
    layout = MappingLayout.from_seed(args.layout_seed) if args.layout_seed is not None else None
    result = MappingMissionSimulation(args.output, args.scenario, layout=layout,
                                      allocator=allocator).run(args.max_ticks)
    print(f"Allocator: {allocator.allocator_id}")
    print(json.dumps(asdict(result), indent=2))
    print(f"Replay: {args.output.resolve() / 'replay.html'}")
    if not result.completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
