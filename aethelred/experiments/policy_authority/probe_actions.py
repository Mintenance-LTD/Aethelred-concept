"""Probe what action a trained policy chooses per opposition type.

Usage: python probe_actions.py <config.yaml> <checkpoint.pt>
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch  # noqa: E402

from aethelred.config.settings import AethelredConfig  # noqa: E402
from aethelred.simulation.environment import AethelredEnv  # noqa: E402
from aethelred.tactical_ai.policy import TacticalPolicy  # noqa: E402
from aethelred.utils.seeding import set_seed  # noqa: E402

ACTION = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    set_seed(0)
    p = TacticalPolicy.build(encoder_config=config.state_encoder,
                             transformer_config=config.tactical_ai)
    p.set_world_size(config.simulation.battlefield.width, config.simulation.battlefield.height)
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
    p.load_policy_weights(ck["policy_weights"])
    p.set_deterministic(True)

    print("seed | opposition   | step0 action | majority action over 30 steps")
    for seed in range(9000, 9012):
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=seed)
        opp = env.threat_spawner.active_profile.name
        acts = []
        for _ in range(30):
            st = env.get_current_state()
            if st is None:
                break
            d = p.decide(st)
            a = d.actions[0].action_type.value if d.actions else "?"
            acts.append(a)
            env.step({"action_type": ACTION.index(a) if a in ACTION else 0,
                      "target_position": [0.5, 0.5], "priority": [0.5], "formation": 0, "target_index": 0})
        maj = Counter(acts).most_common(1)[0]
        print(f"{seed} | {opp:12s} | {acts[0]:12s} | {maj[0]} ({maj[1]}/30)")
        env.close()


if __name__ == "__main__":
    main()
