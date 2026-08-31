# Experiment: can the tactical policy learn state-dependent tactics?

This experiment tests whether the Decision-Transformer + PPO policy can learn a
**state-conditional** tactic — not just a good constant action. It is the basis
for the "does the AI actually learn tactics?" investigation.

## Design

- **Policy authority:** engage units fight only when *commanded* (no hardcoded
  auto-prosecute), and comms-loss autonomy is disabled (`autonomy_enabled: false`),
  so the learned policy is the sole driver.
- **State-dependent opposition (`random_mix`):** each episode is randomly either
  - `exp_winnable` — short-range small-arms the swarm outranges → **engaging wins**, or
  - `exp_deadly` — long-range AA missiles → **withdrawing wins**.
  The threat type is present in the observation, so the right action is a trivial
  function of the state.
- **Reward** rewards kills and survival and penalizes losses (`config.yaml`).
- **Evaluation** is on *held-out* seeds (9000+), disjoint from training (42+),
  against honest baselines: random, every constant action, and a hand-coded
  **oracle** (engage small-arms / retreat from AA).

Validated before training (held-out, 30 seeds):

| Agent | Reward | State-dependent? |
|---|---|---|
| Oracle | **20.2** | yes |
| Best constant (always retreat) | 13.5 | no |

So a learner that reads the threat type should score ~20, well above any constant.

## Result: it does not learn the tactic

Across **10 training configurations** — policy authority, gross-kill reward,
NaN-hardened PPO, reward normalization, a winnable→mixed curriculum, and an
auxiliary representation loss — the policy **always collapsed to a single
state-blind action** (always-engage, always-hold, always-recon, …). Every trained
policy scored ≤ 1.4 and chose the *same* action on winnable and deadly states
(30/30 in the probe).

### Why (diagnosed, not guessed)

- `diag_value.py` showed the value function gave **identical** value to winnable
  and deadly states (`V = −0.222` vs `−0.223`). Under a constant policy both
  contexts genuinely have the same on-policy value, so nothing ever pushes the
  value/policy to distinguish them — a self-reinforcing collapse.
- The **auxiliary loss** (force the encoder to predict the threat composition)
  made the value function separate the contexts (`V = 0.307` vs `0.283`) — i.e.
  it fixed the representation — **and the policy still collapsed** (always-recon).
  This isolates the failure to **policy optimization**, independent of representation:
  vanilla PPO with undirected entropy exploration cannot break out of the
  state-blind equilibrium.

### Contrast demo (`tabular_q.py`): it's temporal credit assignment, not conditioning

Running tabular Q-learning on the *identical* task (same env / reward / held-out
seeds / global-action lever), with the state abstracted to the set of threat
types present:

| Method | Learned policy | Held-out reward |
|---|---|---|
| DT + PPO (10 configs) | state-blind constant | ≤ 1.4 |
| Step-wise tabular Q (per-step action) | retreat *everywhere* | 13.5 (best constant) |
| **Episode-posture bandit (one action/episode)** | **engage small-arms / retreat AA** | **20.2 = oracle** |

The episode-level bandit learns the exact oracle policy with trivial tabular
updates. The step-wise learner — like PPO — collapses to the safe constant.

### Conclusion

The **conditioning is trivially learnable** (the bandit hits the oracle, 20.2). The
real difficulty is **temporal credit assignment**: engaging is a multi-step
commitment whose payoff is delayed, so *per-step* value learners (PPO and step-wise
Q alike) under-credit it and collapse to the immediately-safe action. The fix is
not more PPO tuning — it is matching the control granularity / method to the
decision (episode/posture-level actions, temporal abstraction / options, or an
algorithm robust to delayed reward). For tactical posture decisions like this, the
heavyweight per-step DT+PPO is the wrong tool.

Reproduce the contrast:

```bash
python experiments/policy_authority/tabular_q.py experiments/policy_authority/config.yaml 150
```

## Reproduce (run from the `aethelred/` directory)

```bash
# Train (pure PPO, no online adaptation)
python scripts/train.py --config experiments/policy_authority/config.yaml \
    --episodes 300 --no-noise --no-adapt

# Baselines + trained policy on held-out seeds
python experiments/policy_authority/eval_agents.py experiments/policy_authority/config.yaml \
    9000 30 ckpt:checkpoints/policy_authority/best_policy.pt

# What action does it pick per state type?
python experiments/policy_authority/probe_actions.py experiments/policy_authority/config.yaml \
    checkpoints/policy_authority/best_policy.pt

# Does the value function separate the contexts?
python experiments/policy_authority/diag_value.py experiments/policy_authority/config.yaml \
    checkpoints/policy_authority/best_policy.pt
```
