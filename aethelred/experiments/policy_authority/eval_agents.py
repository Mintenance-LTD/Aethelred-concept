"""Evaluate agents on the random-mix scenario across held-out seeds.

Agents: random, constant:<ACTION>, oracle (engage small-arms / retreat from AA),
ckpt:<path> (trained policy, deterministic). Reports mean reward / survival / kills,
broken down by the per-episode opposition (winnable vs deadly).

Usage: python eval_agents.py <config.yaml> <seed_start> <seed_count> [ckpt:PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch  # noqa: E402

from aethelred.config.settings import AethelredConfig  # noqa: E402
from aethelred.core.enums import EngagementOutcome, ThreatType  # noqa: E402
from aethelred.simulation.environment import AethelredEnv  # noqa: E402
from aethelred.tactical_ai.policy import TacticalPolicy  # noqa: E402
from aethelred.utils.seeding import set_seed  # noqa: E402

ACTION_NAMES = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]
IDX = {n: i for i, n in enumerate(ACTION_NAMES)}


def make_policy(config, ckpt_path):
    set_seed(0)
    p = TacticalPolicy.build(encoder_config=config.state_encoder,
                             transformer_config=config.tactical_ai, device="cpu")
    p.set_world_size(config.simulation.battlefield.width, config.simulation.battlefield.height)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        p.load_policy_weights(ck["policy_weights"])
    p.set_deterministic(True)
    return p


def choose_action(agent, state, env, policy, rng):
    at = 0
    if agent == "random":
        at = int(rng.integers(len(ACTION_NAMES)))
    elif agent.startswith("constant:"):
        at = IDX[agent.split(":", 1)[1]]
    elif agent == "oracle":
        has_aa = any(t.threat_type == ThreatType.AA_MISSILE for t in state.active_threats)
        at = IDX["retreat"] if has_aa else IDX["engage"]
    elif agent == "ckpt":
        decision = policy.decide(state)
        if decision.actions:
            at = IDX.get(decision.actions[0].action_type.value, 0)
            tp = decision.actions[0].target_position
            if tp is not None:
                w, h = env.config.battlefield.width, env.config.battlefield.height
                return {"action_type": at, "target_position": np.array([tp.x / w, tp.y / h], np.float32),
                        "priority": np.array([0.5], np.float32), "formation": 0, "target_index": 0}
    return {"action_type": at, "target_position": np.array([0.5, 0.5], np.float32),
            "priority": np.array([0.5], np.float32), "formation": 0, "target_index": 0}


def run_episode(agent, config, seed, policy):
    env = AethelredEnv(config=config.simulation)
    env.reset(seed=seed)
    opposition = env.threat_spawner.active_profile.name
    rng = np.random.default_rng(seed + 777)
    total = config.simulation.swarm.total_units
    ep_reward = 0.0
    for _ in range(config.simulation.max_steps):
        state = env.get_current_state()
        if state is None:
            break
        ga = choose_action(agent, state, env, policy, rng)
        _, r, term, trunc, info = env.step(ga)
        ep_reward += r
        if term or trunc:
            break
    kills = sum(1 for e in env.get_all_engagements() if e.outcome == EngagementOutcome.TARGET_DESTROYED)
    survival = info["alive_drones"] / total
    env.close()
    return ep_reward, survival, kills, opposition


def evaluate(agent, config, seeds, policy=None):
    rows = {"exp_winnable": [], "exp_deadly": []}
    allr = []
    for s in seeds:
        rew, surv, kills, opp = run_episode(agent, config, s, policy)
        rows.setdefault(opp, []).append((rew, surv, kills))
        allr.append((rew, surv, kills))

    def mean(xs, i):
        return float(np.mean([x[i] for x in xs])) if xs else 0.0

    return {
        "reward": mean(allr, 0), "survival": mean(allr, 1), "kills": mean(allr, 2),
        "win_reward": mean(rows["exp_winnable"], 0), "win_surv": mean(rows["exp_winnable"], 1),
        "win_kills": mean(rows["exp_winnable"], 2),
        "dead_reward": mean(rows["exp_deadly"], 0), "dead_surv": mean(rows["exp_deadly"], 1),
        "n_win": len(rows["exp_winnable"]), "n_dead": len(rows["exp_deadly"]),
    }


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    seed_start, seed_count = int(sys.argv[2]), int(sys.argv[3])
    seeds = list(range(seed_start, seed_start + seed_count))
    ckpt = next((a.split(":", 1)[1] for a in sys.argv[4:] if a.startswith("ckpt:")), None)

    agents = ["random", "constant:engage", "constant:retreat", "constant:evade", "constant:hold", "oracle"]
    policy = None
    if ckpt:
        policy = make_policy(config, ckpt)
        agents.append("ckpt")

    print(f"seeds {seed_start}..{seed_start+seed_count-1}  (n={seed_count})")
    print(f"{'agent':18s} | {'reward':>7s} {'surv':>5s} {'kills':>5s} | "
          f"{'WIN rew/surv/kill':>20s} | {'DEAD rew/surv':>14s}")
    print("-" * 86)
    for agent in agents:
        m = evaluate(agent, config, seeds, policy)
        label = "TRAINED" if agent == "ckpt" else agent
        print(f"{label:18s} | {m['reward']:7.1f} {m['survival']:5.0%} {m['kills']:5.1f} | "
              f"{m['win_reward']:6.1f}/{m['win_surv']:3.0%}/{m['win_kills']:4.1f}      | "
              f"{m['dead_reward']:6.1f}/{m['dead_surv']:3.0%}  (n_win={m['n_win']}, n_dead={m['n_dead']})")


if __name__ == "__main__":
    main()
