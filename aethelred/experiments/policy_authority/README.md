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

### Conclusion

For a tactical decision like this, the DT + single-step-PPO machinery is the wrong
tool — it caps at constants. A lighter method with explicit per-state values and
per-state exploration (e.g. tabular Q-learning / DQN) is expected to learn the
conditional trivially.

## Re-run after the algorithm-audit fixes

The audit (`../../../ALGORITHM_AUDIT.md`) identified several signal-path defects
that could have caused or masked the collapse. All were fixed and the experiment
re-run (pure PPO, `--no-adapt`, 150 episodes; config now uses cross-episode
batches via `update_interval: 512`, `center_advantages: false`, and a short LR
warmup):

- **C2** — the PPO objective now scores only `action_type` (the factor the env
  consumes), not the inert `formation`/`target_index`.
- **C4** — rollouts accumulate across episodes so a PPO batch mixes the winnable
  and deadly contexts, and per-batch advantage mean-centering (which deletes the
  cross-context signal) is disabled.
- **H2** — GAE now bootstraps time-limit truncations with V(s_{t+1}) instead of
  forcing the horizon value to zero.

**Result: the collapse does not break — but it improves.** The trained policy is
still a state-blind constant (the probe shows `retreat` on 30/30 winnable *and*
deadly states), and on held-out seeds it scores exactly the best constant:

| Agent | Reward | Winnable rew/surv/kills | Deadly rew/surv |
|---|---|---|---|
| Oracle | **20.2** | 33.2 / 92% / 8.9 | 7.2 / 79% |
| **TRAINED (audit-fixed PPO)** | **13.5** | 19.7 / 100% / 0.0 | 7.2 / 79% |
| constant: always-retreat | 13.5 | 19.7 / 100% / 0.0 | 7.2 / 79% |

So the fixes moved PPO from a *bad* constant (the original runs scored ≤ 1.4) to
the *best* constant (13.5), but not to a state-dependent policy. This **corroborates
and strengthens** the original conclusion: the collapse survives fixing the
signal-path defects — it is a property of undirected-exploration policy
optimization on this task, not of a broken reward/advantage pipeline.

## The counter bank recovers the conditional PPO can't

The post-audit "divine adaptive engine" (`adaptation/counter_bank.py`) is the
remedy for exactly this failure: it is non-parametric outcome memory, not a
policy-gradient learner. Fed the same `(threat, action, reward)` tuples the
trainer already produces, it recovers the oracle mapping trivially and, followed
as a policy, **matches the oracle** on held-out seeds:

```
$ python experiments/policy_authority/counter_bank_demo.py experiments/policy_authority/config.yaml
Learned counters (from real episode outcomes):
  small_arms       -> engage   (confidence 0.74, turns 7600)
  anti_air_missile -> retreat  (confidence 0.52, turns 8400)

Held-out eval (seeds 9000..9029):
  counter-bank agent | reward  20.2  surv  85%  kills  4.5      # == oracle
```

This is the intended division of labour: PPO optimizes the neural trunk toward a
robust default; the counter bank supplies the explicit per-state values and
per-state exploration that PPO lacks, and the Simple-Domain shield keeps
exploration non-suicidal against threats the bank has not yet mastered.

## Follow-up: can we *hand* PPO the signal? (it fights it)

If the collapse were caused by undirected exploration alone, injecting the
counter bank's recommendation as a per-state action-type **logit prior** — so the
rollouts are already state-dependent — should let PPO learn the conditional. It
does not. `prior_guided.py` pre-trains the bank, then trains PPO with the prior
added to the action-type logits at behaviour time (and re-applied in the update
so importance sampling stays consistent), and evaluates two policies:

```
$ python experiments/policy_authority/prior_guided.py experiments/policy_authority/config.yaml 120 6.0
Counter bank pre-trained:  small_arms -> engage (0.74),  anti_air_missile -> retreat (0.52)

Held-out eval (seeds 9000..9029, prior_scale=6.0):
  COMBINED (net+prior)   | reward  -0.4  surv 63%  kills 0.0 | winnable->relay  deadly->retreat
  NET-ALONE (prior off)  | reward -17.5  surv 34%  kills 0.0 | winnable->relay  deadly->relay
```

The prior pushes `engage` on winnable states with a +4.4 logit, yet the network
learns a `relay` logit large enough to **override** it: COMBINED picks `relay` on
winnable (0 kills) and only retreats on deadly *because the prior forces it*.
Strip the prior (NET-ALONE) and the network is a pure `relay` constant — it
absorbed none of the guidance. Handing PPO the answer made it *worse* than the
no-prior baseline (13.5 → −0.4).

**Why:** for PPO to move the network toward `relay` while its rollouts were
engaging (via the prior), the per-step *advantage* of engaging must have been ≤
that of relaying — even though engaging wins the episode. On winnable states a
kill lands on only ~4% of steps (8.9 kills / 200 steps); the other 96% of engage
steps carry combat risk under a harsh `loss_penalty: 3.0` and no immediate
reward, so the value baseline makes them look locally bad. This is a **per-step
credit-assignment** failure, not an information or exploration failure: the
discriminating variable is in the observation, in the value function (with the
aux loss), and now in the behaviour policy, and PPO *still* collapses. *(This
credit-assignment reading is revised by the shaping test below, which shows a
dense reward does not help — the cause is the constant-vs-conditional collapse
itself.)*

That is exactly why the non-parametric counter bank succeeds where every PPO
variant fails: it estimates the return of each `(threat, action)` pair directly
from many episode outcomes.

## The shaping test: credit assignment was NOT the (whole) story

The prior-guided section proposed per-step credit assignment as the cause and a
dense shaping reward as the test. `config_shaped.yaml` adds `threat_damage: 3.0`
— a per-step reward for damage dealt to threats, which densifies the sparse kill
signal (kills land on ~4% of engage steps → damage lands on most of them). It
does **not** encode the answer: baselines under this reward confirm the premise
holds (constant-engage on deadly still scores −27, oracle 28.3 ≫ best constant):

```
$ python experiments/policy_authority/eval_agents.py experiments/policy_authority/config_shaped.yaml 9000 20
constant:engage | reward  8.0 | WIN 60.9/93%/9.0 | DEAD -27.3/14%
constant:retreat| reward 12.0 | WIN 20.0/100%/0  | DEAD   6.6/78%
oracle          | reward 28.3 | WIN 60.9/93%/9.0 | DEAD   6.6/78%
```

Re-running pure PPO with this dense reward — **the collapse still does not
break.** The probe shows `retreat` on 30/30 winnable *and* deadly states, and:

```
TRAINED (dense-shaped PPO) | reward 12.0 | WIN 20.0/100%/0 | DEAD 6.6/78%   == always-retreat
```

So densifying the signal did **not** rescue PPO — it converges to the same
state-blind `retreat` constant. **This refutes credit assignment as the primary
cause.** The real mechanism is simpler and unifies every run here: a single
state-blind-ish policy, optimized by policy gradient, converges to the best
*constant* action — and `retreat` (safe in both contexts, 12.0) beats `engage`
(8.0, dragged down by deadly losses) *as a constant*. Making winnable-engagement
more lucrative does not change which constant wins, because always-engage is
still worse overall. To beat the best constant the policy must *condition* on the
threat, and that is precisely the step every PPO variant here fails to take —
whether the discriminating variable is in the observation, the value function
(aux loss), the reward density, or the behaviour policy (prior).

**The through-line:** PPO marginalizes over the two contexts inside one policy and
lands on the best constant. The counter bank never marginalizes — it keeps a
*separate* value per `(threat, action)` by construction — so the conditional is
trivial for it (oracle-level) and impossible for PPO here. The lesson from the
whole investigation: on this task the bottleneck is the policy's failure to
*represent and act on* the context split, not information, exploration, or
credit-assignment density.

*(An earlier draft of this file diagnosed the prior-guided result as per-step
credit assignment; the shaping experiment above corrected that — the dense reward
did not help, so the cause is the constant-vs-conditional collapse itself.)*

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

# The counter bank recovers the conditional PPO can't (learns from real outcomes,
# then matches the oracle on held-out seeds)
python experiments/policy_authority/counter_bank_demo.py experiments/policy_authority/config.yaml

# Hand PPO the signal as a logit prior — it fights it (COMBINED vs NET-ALONE)
python experiments/policy_authority/prior_guided.py experiments/policy_authority/config.yaml 120 6.0

# Dense shaping reward — still collapses to always-retreat (train, then probe/eval)
python scripts/train.py --config experiments/policy_authority/config_shaped.yaml \
    --episodes 150 --no-noise --no-adapt
python experiments/policy_authority/probe_actions.py experiments/policy_authority/config_shaped.yaml \
    $(ls -t checkpoints/policy_authority_shaped/checkpoint_ep*.pt | head -1)
```

> Note: probe/eval the *latest* `checkpoint_ep*.pt`, not `best_policy.pt` —
> checkpoint selection is by noisy single-episode reward (audit M3), so
> `best_policy.pt` is often an early winnable-episode snapshot.
