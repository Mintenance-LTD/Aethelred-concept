"""Diagnose whether the value function / policy distribution separate the two
opposition contexts. The key indicator of representational vs policy collapse.

Usage: python diag_value.py <config.yaml> <checkpoint.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
import torch.nn.functional as F

from aethelred.config.settings import AethelredConfig
from aethelred.learning.trainer import ValueHead
from aethelred.simulation.environment import AethelredEnv
from aethelred.tactical_ai.policy import TacticalPolicy
from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder
from aethelred.utils.seeding import set_seed

ACTION = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    set_seed(0)
    p = TacticalPolicy.build(encoder_config=config.state_encoder,
                             transformer_config=config.tactical_ai)
    p.set_world_size(config.simulation.battlefield.width, config.simulation.battlefield.height)
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
    p.load_policy_weights(ck["policy_weights"])
    vh = ValueHead(input_dim=p.config.state_embed_dim)
    has_v = "value_head" in ck
    if has_v:
        vh.load_state_dict(ck["value_head"])
        vh.eval()

    def gather(opp_name):
        Vs, probs = [], []
        for seed in range(9000, 9050):
            env = AethelredEnv(config=config.simulation)
            env.reset(seed=seed)
            if env.threat_spawner.active_profile.name != opp_name:
                env.close()
                continue
            st = env.get_current_state()
            obs = BattlefieldStateEncoder.obs_to_tensors(p._state_to_obs(st), p.device)
            with torch.no_grad():
                emb = p.state_encoder(obs)
                pr = F.softmax(p.forward_step(emb)["action_type_logits"], dim=-1).squeeze(0).numpy()
                v = vh(emb).item() if has_v else float("nan")
            Vs.append(v)
            probs.append(pr)
            env.close()
            if len(Vs) >= 8:
                break
        return float(np.mean(Vs)), np.mean(probs, axis=0)

    for opp in ["exp_winnable", "exp_deadly"]:
        v, pr = gather(opp)
        top = sorted(zip(ACTION, pr), key=lambda x: -x[1])[:3]
        print(f"{opp:13s}  V(s)={v:7.3f}  top-actions: " + ", ".join(f"{a}={x:.2f}" for a, x in top))


if __name__ == "__main__":
    main()
