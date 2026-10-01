"""Run a civilian mapping mission and save its complete audit replay."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aethelred.simulation.mapping_mission import SCENARIOS, MappingMissionSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="unit_loss")
    parser.add_argument("--output", type=Path, required=True, help="Empty directory for mission evidence")
    parser.add_argument("--max-ticks", type=int, default=180)
    args = parser.parse_args()
    result = MappingMissionSimulation(args.output, args.scenario).run(args.max_ticks)
    print(json.dumps(asdict(result), indent=2))
    print(f"Replay: {args.output.resolve() / 'replay.html'}")
    if not result.completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
