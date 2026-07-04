"""Demonstrate that the CounterBank learns the state-dependent tactic that PPO
collapses away from — directly from real episode outcomes.

Pure PPO on this task converges to a state-blind constant (see README): the best
it finds is "always retreat" (~13.5), well short of the oracle (~20.2) that
engages the winnable small-arms and retreats from the deadly AA. The counter bank
is non-parametric outcome memory; fed the same (threat, action, reward) tuples the
trainer already produces, it recovers the conditional trivially.

Phase 1 (learn): play exploratory episodes on TRAIN seeds, feeding per-step
outcomes into a CounterBank exactly as scripts/train.py does.
Phase 2 (report): print the learned counter per threat type.
Phase 3 (eval): an agent that simply follows bank.recommend() on HELD-OUT seeds,
compared against the oracle.

Usage: python counter_bank_demo.py <config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aethelred.adaptation.counter_bank import CounterBank  # noqa: E402
from aethelred.config.settings import AethelredConfig  # noqa: E402
from aethelred.core.enums import EngagementOutcome, ThreatType  # noqa: E402
from aethelred.simulation.environment import AethelredEnv  # noqa: E402

ACTION = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]
IDX = {n: i for i, n in enumerate(ACTION)}
# The exploration set the bank chooses among: the two tactically meaningful ends.
EXPLORE = [IDX["engage"], IDX["retreat"]]


def _gym(at: int) -> dict:
    return {"action_type": at, "target_position": np.array([0.5, 0.5], np.float32),
            "priority": np.array([0.5], np.float32), "formation": 0, "target_index": 0}


def learn(config, seeds) -> CounterBank:
    """Play exploratory episodes, turning the counter bank's wheel from outcomes."""
    bank = CounterBank(min_turns=5)
    for s in seeds:
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=s)
        rng = np.random.default_rng(s)
        for _ in range(config.simulation.max_steps):
            st = env.get_current_state()
            if st is None:
                break
            present = list({t.threat_type for t in st.active_threats})
            at = int(EXPLORE[int(rng.integers(len(EXPLORE)))])
            _, r, term, trunc, _ = env.step(_gym(at))
            for tt in present:
                bank.record_outcome(tt, at, r)
            if term or trunc:
                break
        env.close()
    return bank


def eval_bank(config, seeds, bank) -> dict:
    rows = {"exp_winnable": [], "exp_deadly": []}
    for s in seeds:
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=s)
        opp = env.threat_spawner.active_profile.name
        total = config.simulation.swarm.total_units
        ep_reward, info = 0.0, {}
        for _ in range(config.simulation.max_steps):
            st = env.get_current_state()
            if st is None:
                break
            present = [t.threat_type for t in st.active_threats]
            at = IDX["retreat"]  # safe default when the wheel has not turned enough
            if present:
                idx, conf = bank.recommend(present[0])
                if idx is not None and conf > 0.0:
                    at = idx
            _, r, term, trunc, info = env.step(_gym(at))
            ep_reward += r
            if term or trunc:
                break
        kills = sum(1 for e in env.get_all_engagements()
                    if e.outcome == EngagementOutcome.TARGET_DESTROYED)
        rows.setdefault(opp, []).append((ep_reward, info.get("alive_drones", 0) / total, kills))
        env.close()

    def mean(xs, i):
        return float(np.mean([x[i] for x in xs])) if xs else 0.0

    allr = rows["exp_winnable"] + rows["exp_deadly"]
    return {
        "reward": mean(allr, 0), "surv": mean(allr, 1), "kills": mean(allr, 2),
        "win": (mean(rows["exp_winnable"], 0), mean(rows["exp_winnable"], 1), mean(rows["exp_winnable"], 2)),
        "dead": (mean(rows["exp_deadly"], 0), mean(rows["exp_deadly"], 1)),
        "n_win": len(rows["exp_winnable"]), "n_dead": len(rows["exp_deadly"]),
    }


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    bank = learn(config, range(42, 122))  # TRAIN seeds (disjoint from eval)

    print("Learned counters (from real episode outcomes):")
    for tt in (ThreatType.SMALL_ARMS, ThreatType.AA_MISSILE):
        idx, conf = bank.recommend(tt)
        act = ACTION[idx] if idx is not None else "-"
        print(f"  {tt.value:16s} -> {act:8s} (confidence {conf:.2f}, turns {bank.turns(tt)})")

    m = eval_bank(config, range(9000, 9030), bank)  # HELD-OUT seeds
    print("\nHeld-out eval (seeds 9000..9029):")
    print(f"  counter-bank agent | reward {m['reward']:5.1f}  surv {m['surv']:4.0%}  kills {m['kills']:4.1f}")
    print(f"    winnable: rew {m['win'][0]:5.1f} / surv {m['win'][1]:4.0%} / kills {m['win'][2]:4.1f}"
          f"   deadly: rew {m['dead'][0]:5.1f} / surv {m['dead'][1]:4.0%}"
          f"   (n_win={m['n_win']}, n_dead={m['n_dead']})")
    print("\n(Compare: pure-PPO trained policy = 13.5 = always-retreat; oracle = 20.2.)")


if __name__ == "__main__":
    main()
