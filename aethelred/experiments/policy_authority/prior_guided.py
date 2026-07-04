"""Does handing PPO the per-state signal (as a logit prior) break the collapse?

Pure PPO on this task collapses to a state-blind constant because the discriminating
variable (threat type) gets marginalised away by undirected exploration — even
though it is in the observation and (with the aux loss) in the value function.

This experiment hands PPO the answer as *guided exploration*: the counter bank's
recommendation for the current threat is added to the action-type logits at
behaviour time (and re-applied in the PPO update, so importance sampling stays
consistent). The rollouts are therefore state-dependent, and the policy gradient
has a real per-context signal to reinforce.

We then measure two things on held-out seeds:
  1. COMBINED  = network + prior            -> is the deployed policy conditional?
  2. NET-ALONE = network with the prior off -> did the guidance transfer into the
     weights, or is the prior a permanent crutch?

Usage: python prior_guided.py <config.yaml> [episodes] [prior_scale]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch  # noqa: E402

from aethelred.adaptation.counter_bank import CounterBank  # noqa: E402
from aethelred.config.settings import AethelredConfig  # noqa: E402
from aethelred.core.enums import EngagementOutcome  # noqa: E402
from aethelred.learning.trainer import PPOTrainer  # noqa: E402
from aethelred.simulation.environment import AethelredEnv  # noqa: E402
from aethelred.tactical_ai.policy import TacticalPolicy  # noqa: E402
from aethelred.tactical_ai.state_encoder import BattlefieldStateEncoder  # noqa: E402
from aethelred.utils.seeding import set_seed  # noqa: E402

ACTION = ["move", "engage", "evade", "formation_change", "recon", "relay", "retreat", "hold"]
IDX = {n: i for i, n in enumerate(ACTION)}
EXPLORE = [IDX["engage"], IDX["retreat"]]
NUM_ACTIONS = len(ACTION)


def _gym(at: int) -> dict:
    return {"action_type": at, "target_position": np.array([0.5, 0.5], np.float32),
            "priority": np.array([0.5], np.float32), "formation": 0, "target_index": 0}


def pretrain_bank(config, seeds, scale: float) -> CounterBank:
    """Populate the counter bank from exploratory play (as counter_bank_demo)."""
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
    bank._prior_scale = scale  # stash for prior_for()
    return bank


def prior_for(bank, state, scale: float) -> np.ndarray:
    """Additive action-type logit prior = the bank's recommendation for the
    dominant present threat (zeros if none / untrusted)."""
    prior = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if state.active_threats:
        tt = state.active_threats[0].threat_type
        prior += bank.action_prior(tt, scale=scale).astype(np.float32)
    return prior


def train(config, bank, episodes: int, scale: float) -> TacticalPolicy:
    set_seed(42)
    policy = TacticalPolicy.build(config.state_encoder, config.tactical_ai, device="cpu")
    policy.set_world_size(config.simulation.battlefield.width, config.simulation.battlefield.height)
    trainer = PPOTrainer(policy, config.training, checkpoint_dir="checkpoints/prior_guided",
                         log_dir="runs/prior_guided")
    for ep in range(episodes):
        env = AethelredEnv(config=config.simulation)
        env.reset(seed=42 + ep)  # TRAIN seeds
        for step in range(config.simulation.max_steps):
            st = env.get_current_state()
            if st is None:
                break
            prior = prior_for(bank, st, scale)
            obs_t = BattlefieldStateEncoder.obs_to_tensors(env._state_to_obs(st), trainer.device)
            ga = trainer.select_action(obs_t, action_prior=prior)
            _, r, term, trunc, _ = env.step(ga)
            done = term or trunc
            boot = 0.0
            if trunc and not term:
                boot = trainer.estimate_value(
                    BattlefieldStateEncoder.obs_to_tensors(env._state_to_obs(env.get_current_state()), trainer.device)
                )
            trainer.finish_step(r, done, terminated=term, bootstrap_value=boot)
            if trainer.ready_to_update:
                trainer.train_on_rollout()
            if done:
                break
        env.close()
    if len(trainer.rollout_buffer) >= 16:
        trainer.train_on_rollout()
    trainer.close()
    return policy


@torch.no_grad()
def decide_action(policy, state, bank, scale: float, use_prior: bool) -> int:
    obs = BattlefieldStateEncoder.obs_to_tensors(policy._state_to_obs(state), policy.device)
    logits = policy.forward_step(policy.state_encoder(obs))["action_type_logits"]
    if use_prior:
        logits = logits + torch.as_tensor(prior_for(bank, state, scale))
    return int(logits.argmax(dim=-1).item())


def evaluate(policy, config, seeds, bank, scale: float, use_prior: bool) -> dict:
    rows = {"exp_winnable": [], "exp_deadly": []}
    actions_by_ctx = {"exp_winnable": Counter(), "exp_deadly": Counter()}
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
            at = decide_action(policy, st, bank, scale, use_prior)
            actions_by_ctx.setdefault(opp, Counter())[ACTION[at]] += 1
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
        "win": (mean(rows["exp_winnable"], 0), mean(rows["exp_winnable"], 2)),
        "dead": (mean(rows["exp_deadly"], 0), mean(rows["exp_deadly"], 1)),
        "win_act": actions_by_ctx["exp_winnable"].most_common(1),
        "dead_act": actions_by_ctx["exp_deadly"].most_common(1),
    }


def _report(tag, m):
    wa = m["win_act"][0][0] if m["win_act"] else "-"
    da = m["dead_act"][0][0] if m["dead_act"] else "-"
    print(f"  {tag:22s} | reward {m['reward']:5.1f}  surv {m['surv']:4.0%}  kills {m['kills']:4.1f}"
          f"  | winnable->{wa:8s} deadly->{da:8s}")


def main():
    config = AethelredConfig.from_yaml(sys.argv[1])
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    from aethelred.core.enums import ThreatType

    bank = pretrain_bank(config, range(42, 122), scale)
    print("Counter bank pre-trained. Learned counters:")
    for tt in (ThreatType.SMALL_ARMS, ThreatType.AA_MISSILE):
        idx, conf = bank.recommend(tt)
        print(f"  {tt.value:16s} -> {ACTION[idx] if idx is not None else '-':8s} (conf {conf:.2f})")

    policy = train(config, bank, episodes, scale)

    seeds = range(9000, 9030)
    print(f"\nHeld-out eval (seeds 9000..9029, {episodes} train episodes, prior_scale={scale}):")
    _report("COMBINED (net+prior)", evaluate(policy, config, seeds, bank, scale, use_prior=True))
    _report("NET-ALONE (prior off)", evaluate(policy, config, seeds, bank, scale, use_prior=False))
    print("\n(Reference: pure-PPO no-prior = 13.5 = always-retreat; oracle = 20.2.)")
    print("If NET-ALONE is state-dependent (winnable->engage, deadly->retreat), the")
    print("guidance transferred into the weights; if it reverts to a constant, the")
    print("prior is doing the conditioning and PPO still cannot hold the split.")


if __name__ == "__main__":
    main()
