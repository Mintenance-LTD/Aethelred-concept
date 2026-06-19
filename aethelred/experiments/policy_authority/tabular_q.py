"""Contrast demo: a tiny tabular Q-learner on the IDENTICAL random-mix task.

Same env, same opposition, same reward, same held-out seeds, same lever (the
global action type). The ONLY difference from the DT+PPO policy is the method:
an explicit per-state value table with epsilon-greedy per-state exploration.

State abstraction = the set of threat TYPES currently present (derived from the
same observation the neural policy saw). This is a coarse, general abstraction,
not the answer hand-coded in.

Usage: python tabular_q.py <config.yaml> [train_episodes]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aethelred.config.settings import AethelredConfig  # noqa: E402
from aethelred.core.enums import EngagementOutcome  # noqa: E402
from aethelred.simulation.environment import AethelredEnv  # noqa: E402
from aethelred.utils.seeding import set_seed  # noqa: E402

ACTION_NAMES = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]
N_ACTIONS = len(ACTION_NAMES)


def discretize(state):
    """State = the set of threat types present (general, obs-derived)."""
    return frozenset(t.threat_type for t in state.active_threats)


def gym_action(a: int) -> dict:
    return {"action_type": a, "target_position": np.array([0.5, 0.5], np.float32),
            "priority": np.array([0.5], np.float32), "formation": 0, "target_index": 0}


def train(config, episodes: int):
    Q: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
    alpha, gamma = 0.1, 0.99
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=42 + ep)  # training seeds (disjoint from held-out 9000+)
        state = env.get_current_state()
        s = discretize(state)
        for _ in range(config.simulation.max_steps):
            a = int(np.random.randint(N_ACTIONS)) if np.random.random() < eps else int(np.argmax(Q[s]))
            _, r, term, trunc, _ = env.step(gym_action(a))
            ns = env.get_current_state()
            s2 = discretize(ns) if ns is not None else s
            Q[s][a] += alpha * (r + gamma * float(np.max(Q[s2])) - Q[s][a])
            s = s2
            if term or trunc:
                break
        env.close()
    return Q


def train_bandit(config, episodes: int):
    """Episode-level posture bandit: pick ONE action per episode (by threat set),
    hold it the whole episode, update its running-mean episode return. This removes
    the per-step temporal-credit-assignment difficulty."""
    Q: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
    counts: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=42 + ep)
        s = discretize(env.get_current_state())
        a = int(np.random.randint(N_ACTIONS)) if np.random.random() < eps else int(np.argmax(Q[s]))
        ep_r = 0.0
        for _ in range(config.simulation.max_steps):
            if env.get_current_state() is None:
                break
            _, r, term, trunc, _ = env.step(gym_action(a))
            ep_r += r
            if term or trunc:
                break
        counts[s][a] += 1
        Q[s][a] += (ep_r - Q[s][a]) / counts[s][a]  # running mean of episode return
        env.close()
    return Q


def evaluate(config, Q, seeds):
    rows = {"exp_winnable": [], "exp_deadly": []}
    for seed in seeds:
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=seed)
        opp = env.threat_spawner.active_profile.name
        total = config.simulation.swarm.total_units
        ep_r = 0.0
        for _ in range(config.simulation.max_steps):
            st = env.get_current_state()
            if st is None:
                break
            s = discretize(st)
            a = int(np.argmax(Q[s])) if s in Q else ACTION_NAMES.index("hold")
            _, r, term, trunc, info = env.step(gym_action(a))
            ep_r += r
            if term or trunc:
                break
        kills = sum(1 for e in env.get_all_engagements() if e.outcome == EngagementOutcome.TARGET_DESTROYED)
        rows.setdefault(opp, []).append((ep_r, info["alive_drones"] / total, kills))
        env.close()

    def mean(xs, i):
        return float(np.mean([x[i] for x in xs])) if xs else 0.0

    allr = rows["exp_winnable"] + rows["exp_deadly"]
    return {
        "reward": mean(allr, 0), "survival": mean(allr, 1), "kills": mean(allr, 2),
        "win": (mean(rows["exp_winnable"], 0), mean(rows["exp_winnable"], 1), mean(rows["exp_winnable"], 2)),
        "dead": (mean(rows["exp_deadly"], 0), mean(rows["exp_deadly"], 1)),
        "n_win": len(rows["exp_winnable"]), "n_dead": len(rows["exp_deadly"]),
    }


def _report(name, config, Q, episodes):
    print(f"\n=== {name} ({episodes} training episodes) ===")
    print("  learned greedy policy:")
    for s in sorted(Q.keys(), key=lambda x: str(sorted(t.value for t in x))):
        types = ", ".join(sorted(t.value for t in s)) or "(no threats)"
        print(f"    threats={{{types}}}  ->  {ACTION_NAMES[int(np.argmax(Q[s]))]}")
    m = evaluate(config, Q, range(9000, 9030))
    print(f"  held-out: reward={m['reward']:.1f}  survival={m['survival']:.0%}  kills={m['kills']:.1f}  "
          f"| WIN {m['win'][0]:.1f}/{m['win'][1]:.0%}/{m['win'][2]:.1f}k  DEAD {m['dead'][0]:.1f}/{m['dead'][1]:.0%}")


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    set_seed(0)
    _report("STEP-WISE Q (per-step action)", config, train(config, episodes), episodes)

    set_seed(0)
    _report("EPISODE-POSTURE BANDIT (one action/episode)", config, train_bandit(config, episodes), episodes)

    print("\n  reference: oracle 20.2 | best constant retreat 13.5 | DT+PPO collapsed <= 1.4")


if __name__ == "__main__":
    main()
