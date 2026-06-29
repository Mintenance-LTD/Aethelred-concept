# Algorithm audit — Aethelred tactical-AI learning stack

**Scope:** the reinforcement-learning algorithm and its supporting model stack —
the PPO trainer (`learning/trainer.py`), the policy/Decision-Transformer stack
(`tactical_ai/`), the environment learning signal (`simulation/environment.py`,
`tactical_ai/reward_model.py`), and the **policy-authority experiment**
(`experiments/policy_authority/`) that concluded "DT + single-step PPO cannot
learn a state-dependent tactic."

**Headline finding:** the experiment's *measurements* are sound, but its
*diagnosis* is overstated. The setup that "fails" is not really a Decision
Transformer, and the residual failure is a well-understood (and fixable)
exploration / credit-assignment problem — not proof that PPO "caps at constants."
There are also several concrete correctness bugs/smells in the PPO update worth
fixing regardless of the experiment.

Severity legend: 🔴 correctness / affects results · 🟠 design issue that distorts
conclusions · 🟡 smell / maintainability.

---

## 1. The "Decision Transformer" is inert — the failing learner is an MLP 🟠

`TacticalPolicy.forward_step` is the single forward path used by **both**
inference (`decide`) and PPO (`select_action` / `train_on_rollout`):

```python
# tactical_ai/policy.py:84-97
def forward_step(self, state_embed):
    states  = state_embed.unsqueeze(1)                       # (B, 1, D)  -- length-1
    actions = torch.zeros(b, 1, self.config.action_embed_dim)  # zero action token
    rtg     = torch.full((b, 1, 1), float(self._target_return)) # CONSTANT rtg
    timesteps = torch.zeros(b, 1, dtype=torch.long)            # always t=0
    logits = self.transformer(states, actions, rtg, timesteps)
    return {k: (v[:, -1] if v.dim() == 3 else v) ...}
```

Consequences:

- The context length is always **1**. The causal mask, timestep embedding and
  action-token embedding all operate on a single timestep and contribute nothing.
- The history buffers `_state_history`, `_action_history`, `_rtg_history`,
  `_timestep_history` (`policy.py:40-43`) are **declared and cleared but never
  written to and never read**. `decide()` does not append to them; `forward_step`
  ignores them.
- **Return-to-go conditioning is a constant input** (`self._target_return`, default
  `10.0`). A Decision Transformer's whole premise is conditioning behaviour on the
  desired return; here RTG is the same scalar for every state in training and
  inference, so the `return_embed` adds a constant bias and nothing else.
  `update_return()` / `_current_return` (`policy.py:99-101`) is **dead code** —
  the decremented RTG is never fed anywhere.

So the learner that "collapses to a constant" is effectively: *attention-pooling
state encoder → a 6-layer transformer block used as a fixed-context MLP → action
head → single-step PPO.* Attributing the failure to "DT" (README §Conclusion) is
mislabeled — the DT apparatus is inert weight the optimizer must still drag along.
This matters because the recommended alternative ("tabular Q / DQN") would also
remove this inert machinery, so the comparison is not apples-to-apples.

**Recommendation:** either (a) actually feed the history buffers + a decreasing
RTG into `forward_step` so it is a real DT, or (b) drop the DT framing and call it
what it is (an MLP/transformer-block policy). Don't conclude "DT can't" from a
configuration where the DT never runs as a DT.

---

## 2. Only 3 of 5 action factors get policy gradients 🟠

`_logp_entropy` (`trainer.py:397-418`) and `ActionHead.log_prob`
(`action_head.py:130-151`) cover **action_type, formation, target_index** only.
`target_position` and `priority` come from `Sigmoid` heads (`action_head.py:37-46`),
are sampled deterministically, and are **excluded** from the log-prob, the entropy
term, and therefore the PPO objective.

- For tasks where *where to go* / *how hard* matters, those factors are
  structurally unlearnable by this trainer.
- For the policy-authority experiment, position/priority are irrelevant — **but so
  are `formation` (6 options) and `target_index` (16 options)**, and those *are*
  in the objective. The entropy bonus and the importance-ratio variance are
  therefore spent mostly on two irrelevant categorical factors, diluting the
  exploration pressure on the **one** factor that decides winnable-vs-deadly:
  `action_type`. This makes the "undirected exploration can't escape" outcome more
  likely and is a property of the action parameterization, not of PPO.

**Recommendation:** mask/disable the irrelevant factors for the experiment (sample
them as constants), and either make position/priority proper distributions
(e.g. Beta/Gaussian heads with log-probs) or document them as fixed.

---

## 3. The residual collapse is exploration/credit-assignment, not "PPO caps at constants" 🟠

The README's own aux-loss result is the key evidence: forcing the encoder to
predict the threat composition made the **value function separate the contexts**
(`V=0.307` vs `0.283`) yet the **policy still collapsed**. That isolates the
failure to *policy optimization*. The specific setup makes escaping a passive
local optimum hard for reasons that are fixable:

- **Delayed reward vs truncated advantage horizon.** Commanding `ENGAGE` does not
  produce an immediate kill: `_decode_unit_action` converts `ENGAGE` into `MOVE`
  until a unit is within `sensor_range` and has ammo
  (`environment.py:369-383`). So the kill reward (weight `threat_neutralized: 4.0`
  → `×0.5` per kill) lands many steps after the action that earned it. Meanwhile
  PPO updates on **64-step fragments** (`update_interval: 64`) and GAE bootstraps
  the most-recent step of each fragment with `next_value = 0` and treats
  truncation as termination (see §4). Long-delayed credit is repeatedly discarded
  at fragment/episode boundaries — i.e. exactly the signal that would teach
  "engaging paid off in this state" is attenuated.
- **Undirected joint entropy** over an 8×6×16 space, mostly spent on irrelevant
  factors (§2), is weak directed exploration. A passive action (`hold`/`recon`)
  yields the *same* value in both contexts (no kills, few losses), so the value
  baseline gives it ~0 advantage everywhere and there is no representational
  gradient out of it — but that's a generic local-optimum/exploration trap, not a
  ceiling intrinsic to PPO.

**Honest conclusion:** *"this PPO configuration — inert DT, undirected exploration
over mostly-irrelevant action factors, and a truncated advantage horizon on a
delayed-reward task — falls into a passive-action equilibrium."* The
recommendation to try DQN/tabular Q is reasonable, but the stated reason ("DT +
single-step-PPO machinery … caps at constants") generalizes from a degenerate
configuration. Worth testing before concluding: action-factor masking (§2),
larger/whole-episode rollouts with correct bootstrapping (§4), a per-action-type
entropy floor, and reward shaping that pays an immediate signal for the
engage/withdraw decision.

---

## 4. GAE treats truncation as termination 🔴

`train.py:236` collapses both end conditions into one flag:

```python
done = terminated or truncated
trainer.finish_step(reward, done)
```

and `compute_gae` (`trainer.py:73-83`) zeroes the bootstrap whenever `step.done`:

```python
if step.done:
    next_value = 0.0
    gae = 0.0
delta = step.reward * reward_scale + gamma * next_value - step.value
```

A time-limit truncation is **not** a terminal state — its return should bootstrap
`V(s_T)`, not 0. With `max_steps: 200`, essentially every episode ends by
truncation, so the value targets near the end of every episode are systematically
biased low. Additionally, the *most recent* step of every mid-episode 64-step
fragment is bootstrapped with the initial `next_value = 0.0` (it is not `done`,
but there is no later step in the fragment), truncating the advantage horizon at
each update boundary.

**Recommendation:** distinguish `terminated` from `truncated`; store a bootstrap
value for the last non-terminal step of each rollout (re-run the value head on the
final next-observation) and only zero the bootstrap on true termination.

---

## 5. Behaviour policy (collection) uses eval-mode; PPO scores it in train-mode 🔴

The transformer has `dropout=0.1` (`DecisionTransformerConfig.dropout`, applied in
the `TransformerEncoderLayer`). Collection and update run it in **different modes**:

- `select_action` → `self.policy.transformer.eval()` (dropout **off**) when the
  stored `log_prob` is computed (`trainer.py:427-449`).
- `train_on_rollout` → `self.policy.transformer.train()` (dropout **on**) when
  `new_log_probs` and the ratio are computed (`trainer.py:480, 542-556`).

PPO's importance ratio `exp(new_log_prob − old_log_prob)` therefore compares two
*different* stochastic functions even on the first epoch before any weight update,
injecting noise into the ratio and the clipped objective. On-policy methods should
keep the sampling and scoring distributions identical.

**Recommendation:** set transformer `dropout=0.0` for PPO (simplest and standard
for on-policy RL), or compute `old_log_prob` in the same `train()` mode used by the
update.

---

## 6. Reward-normalization "initialized" sentinel is fragile 🟡

```python
# trainer.py:492
self._ret_std = batch_std if self._ret_std == 1.0 else (0.99*self._ret_std + 0.01*batch_std)
```

`_ret_std == 1.0` is overloaded to mean "uninitialized." If the running std ever
legitimately equals `1.0` it will "snap" again to the new batch std instead of
tracking via EMA. Low-probability, but it conflates a sentinel with a real value.

**Recommendation:** use an explicit `self._ret_std_initialized = False` flag.

---

## 7. Dead / misleading components 🟡

- **`RewardModel` is instantiated but unused.** `PPOTrainer.__init__` sets
  `self.reward_model = RewardModel()` (`trainer.py:347`); the environment computes
  reward (`environment.py:_compute_reward`). The class — including
  `compute_return_to_go` — is never exercised by training. Either wire it in or
  remove it; right now two reward definitions exist (`reward_model.py` and
  `environment.py`) and only one is live, which is a trap for future readers.
- **RTG / history machinery** (§1) is dead code that implies a capability the code
  does not use.
- **Tail-of-episode experience is silently dropped.** With `update_interval: 64`
  and `max_steps: 200`, the episode-end flush discards the final `<16`-step
  remainder (`train.py:271-274`). Minor data loss; fine to leave, worth a comment.

---

## 8. What is correct / good (so the audit is balanced) ✅

These were verified and look right:

- **Gradients reach the encoder + transformer.** `train_on_rollout` re-encodes the
  stored raw observations and re-runs `forward_step`, so the policy network (not
  just the value head) is optimized (`trainer.py:534-561`). The earlier "only the
  value head trains" bug is genuinely fixed.
- **Train/inference parity.** Both `decide` and `select_action` go through the same
  `forward_step`, so the deployed policy is the optimized one.
- **GAE math, PPO clipping, value clipping, advantage normalization** are textbook
  and correctly implemented (aside from the truncation issue in §4).
- **NaN hardening is real and correct:** log-ratio clamp before `exp`
  (`trainer.py:556`), non-finite loss/grad skips (`trainer.py:598-615`), and the
  all-masked-row attention guard that prevents a softmax-over-all-`-inf` NaN
  (`state_encoder.py:116-130`).
- **LR schedule** sets each group's LR from a captured base (warmup → cosine),
  fixing the prior "multiply-in-place collapses LR to 0" bug
  (`trainer.py:646-662`).
- **Auxiliary loss target is indexed correctly.** `threats[:, :, 8:14]`
  (`trainer.py:582`) selects exactly the 6-element threat-type one-hot
  (feature layout: 8 scalars + 6 type + 4 category = 18; `core/models.py:136-160`).
- **The experiment's premise holds:** the threat type *is* observable (one-hot +
  `estimated_range`), and the eval harness is honest — held-out seeds (9000+)
  disjoint from training (42+), with random/constant/oracle baselines.
- **Reward/value scaling is internally consistent:** stored `value` and GAE both
  live in the normalized (reward×`reward_scale`) units, so returns/advantages are
  coherent.

---

## Priority order for fixes

1. 🔴 §4 truncation-vs-termination bootstrapping (biases every episode's value
   targets).
2. 🔴 §5 dropout mode mismatch between collection and update (corrupts the PPO
   ratio).
3. 🟠 §2 mask the irrelevant action factors for the experiment, then re-run before
   accepting the "PPO can't" conclusion.
4. 🟠 §1 make the DT real *or* stop calling it a DT; remove the dead RTG/history
   path.
5. 🟠 §3 re-test with whole-episode rollouts + a directed/per-action-type
   exploration bonus and/or an immediate engage/withdraw reward signal.
6. 🟡 §6, §7 sentinel flag, remove/wire `RewardModel`, comment the tail-drop.

**Bottom line:** the prior fixes (gradient flow, LR, parity, NaN-safety, seeding)
are real and the experiment is honestly measured, but the conclusion that
"DT + single-step PPO categorically cannot learn a state-dependent tactic" is not
supported by the code as written — the DT is inert, exploration is undirected over
mostly-irrelevant factors, and the advantage horizon is truncated on a
delayed-reward task. Fix §2/§4/§5 and re-run before treating PPO as the wrong tool.
</content>
</invoke>
