# Algorithm Audit — Aethelred learning stack

**Date:** 2026-07-02
**Scope:** the full learning algorithm — observation builder → state encoder → decision
transformer → action head → PPO trainer → environment/reward → adaptation engine
(MAML/EWC/replay) → learning loop → experiment harness (`experiments/policy_authority/`).
**Method:** line-by-line review of every module in the pipeline, with the highest-impact
findings verified by executing code against the installed package (pydantic 2.13.4,
torch 2.12.1). The existing test suite passes (21 passed), consistent with the README.

The repo's own experiment (`experiments/policy_authority/`) reports that the policy
always collapses to a state-blind constant action and concludes the failure is
fundamental to "DT + single-step PPO". **This audit finds several concrete
implementation defects in the PPO signal path that each plausibly cause or contribute
to that collapse.** The conclusion "PPO caps at constants" is not yet supported —
the machinery being optimized is not the machinery the environment responds to,
and the advantage signal that would distinguish the two contexts is structurally
removed before it reaches the gradient. Details below.

---

## Critical findings

### C1. Mission-progress reward is always zero (verified by execution)

`AethelredEnv._compute_reward` compares `state_before.objectives` against
`state_after.objectives` (`environment.py:436-439`), but both `BattlefieldState`
snapshots hold **references to the same `ObjectiveState` objects** (`_build_state`
passes `self._objectives`; pydantic v2 does not copy nested model instances).
When `step()` sets `obj.is_completed = True`, the "before" state mutates too, so
`new_completions` is always 0.

Verified empirically: with `mission_progress: 100.0`, an objective completed
during a step yields **reward 0.0**.

Compounding it: `_check_terminated` ends the episode when all objectives complete,
which cuts off the per-step survival reward stream. Net effect of the current
composite objective: **completing the mission earns nothing and forfeits future
reward** — the learning signal actively discourages the mission. The same aliasing
applies to drone/threat objects (`state_before` is not a true snapshot), though the
other reward terms happen not to depend on it (survival uses total counts;
kills use engagement events).

*Fix:* snapshot with `model_copy(deep=True)` (or compare against cached counts
taken before mutation), and add a terminal completion bonus that dominates the
foregone survival stream.

### C2. Two of the three PPO action factors are inert in the environment

The PPO joint log-prob and entropy are computed over `action_type`, `formation`,
and `target_index` (`trainer.py:397-418`). But in the training path
(`env.step` → `_decode_action`, `environment.py:267-308`):

- **`target_index` is never read.** Threat assignment is round-robin
  (`active_threats[engage_i % len(active_threats)]`). The pointer head the policy
  samples over 16 targets has zero effect on the world.
- **`formation` is decoded but never consumed** — `apply_decision` and engagement
  resolution ignore `TacticalAction.formation`; the swarm coordinator that does use
  formations is not in the training loop.

Consequences: two-thirds of the factored action space contributes noise to the PPO
importance ratio. As the shared trunk updates, the log-probs of these inert factors
drift, moving `ratio = exp(new_lp − old_lp)` away from 1 for reasons unrelated to
reward, and triggering the PPO clip — which then gates gradients for the one factor
that matters (`action_type`). The entropy bonus (`entropy_coef 0.05` in the
experiment) is also mostly spent maximizing entropy over 16 + 6 categories nothing
responds to.

*Fix:* either wire `target_index`/`formation` into the env, or drop them from the
log-prob/entropy until they matter. Also mask `target_logits` to the number of
active threats.

### C3. Rollout and update run the network in different modes (dropout mismatch)

`select_action` records `old_log_probs` with the encoder/transformer in `.eval()`
(`trainer.py:427-428`), while `train_on_rollout` recomputes `new_log_probs` in
`.train()` (`trainer.py:480-481`) with the transformer's **18 dropout layers
(p=0.1) active**. So even on the first epoch over freshly collected data, the
importance ratio is not 1 and the "on-policy" assumption is broken; each minibatch
sees a differently-dropped subnetwork.

Verified: forward passes in train mode are stochastic and differ from eval mode.
At initialization the effect is tiny (the 0.02-gain init makes the transformer
near-passthrough), but it grows with weight magnitude over training.

*Fix:* the standard practice for PPO is dropout off in both phases
(`dropout=0.0`), or at minimum compute the update-time log-probs in eval mode.

### C4. Every PPO batch is single-context, and advantage normalization then deletes the cross-context signal

The experiment's whole point is a per-episode context (`exp_winnable` vs
`exp_deadly`) that demands a state-dependent response. But:

- `scripts/train.py` flushes the rollout buffer at every episode end
  (`train.py:271-274`), and `update_interval` (64 in the experiment) is well below
  the 200-step episode — so **every PPO update batch comes from exactly one
  episode, i.e. one context**.
- `train_on_rollout` then mean-centers advantages per batch
  (`trainer.py:513-515`). Within a single-context batch, the context-level
  advantage — "everything in a winnable episode outperforms the cross-context
  value baseline" — is a common offset, and mean-centering removes it exactly.

What survives is only within-episode action-to-action contrast, a much weaker and
noisier signal, further degraded by C2/C3 ratio noise. The value function is the
only remaining channel for cross-context information, and its own movement per
update is capped by value clipping (`ppo_clip_ratio 0.2` applied in
normalized-return units) under a drifting reward scale (`_ret_std` EMA). This is a
coherent mechanistic explanation for the observed result (value separates contexts
only weakly even with the aux loss — 0.307 vs 0.283 — and the policy stays
state-blind) that does **not** require "PPO fundamentally can't do this".

*Fix for the experiment:* collect rollouts **across** episode boundaries so each
update mixes both contexts (remove the per-episode flush; let `update_interval`
span ≥2 episodes), and/or skip per-batch advantage mean-centering (keep std
scaling) when batches are context-homogeneous.

### C5. In default mode, the adaptation engine and PPO fight over the same weights

`learning_loop.maybe_adapt()` (called every step in non-`--no-adapt` training)
overwrites the transformer's weights mid-episode (`learning_loop.py:112-127`).
PPO's rollout buffer still holds `old_log_probs` sampled under the pre-adaptation
policy, and the AdamW moments in `trainer.optimizer` refer to parameters that were
just replaced behind the optimizer's back. Every adaptation invalidates the
on-policy data and the optimizer state.

Also note what the adaptation actually is: `_counter_action_index`
(`adaptation_engine.py:281-293`) is a **hand-written rule table**
(EW → formation change, laser/close ambush → retreat, else evade). The
`ThreatResponseHead` is trained by supervised learning to reproduce that table,
and 10% of its final bias is blended into the policy's `action_type_head` bias
(`_transfer_adaptation_to_model`). This is a hardcoded heuristic nudge, not
learning from outcomes — worth stating plainly in the docs, since the README
describes it as "learns a real threat→counter mapping".

*Fix:* at minimum, flush the rollout buffer after any adaptation write and rebuild
optimizer state; better, route adaptation through the same optimizer/loss as PPO
(e.g. as a shaped auxiliary loss) instead of direct weight writes.

---

## High-priority findings

### H1. The movement target is untrainable

`position_head` and `priority_head` outputs are deterministic and excluded from
the log-prob ("as in the original design", `trainer.py:403-406`), and no other
loss touches them — so **no gradient ever reaches them**. The commanded target
position (which drives MOVE/RECON/EVADE/hold-position behavior — arguably the most
important continuous output) is frozen at its random-init function of the state,
drifting only as a side effect of trunk updates. If "where to go" should be
learnable, model it as a distribution (e.g. Gaussian or discretized grid) and
include it in the PPO objective.

### H2. Time-limit truncation treated as termination in GAE

`train.py:236` stores `done = terminated or truncated`; `compute_gae` zeroes the
bootstrap at every `done` (`trainer.py:73-77`). With a per-step survival reward
and `max_steps` truncation being the common episode end for passive policies, the
value target at late timesteps is systematically biased low. Standard fix:
bootstrap truncated (not terminated) final steps with `V(s_T)`.

### H3. Train/eval observation parity gap on the time feature

`build_observation`'s time feature is `timestep / time_scale`. The env passes
`time_scale=max_steps` (200 in the experiment, `environment.py:260`), while
`policy.decide` / the eval+diag harnesses use the default `time_scale=1000`
(`state_encoder.py:21`, `policy.py:148-150`). Training and evaluation therefore
disagree 5× on one input dimension — the same class of drift the position
normalization fix was meant to eliminate. (Evaluation also applies no geofence
clamp, while training does, and `eval_agents.py` hardcodes `formation: 0` rather
than the policy's output — harmless today only because of C2.)

### H4. Loss events double-fed to the learning loop

`train.py:254` calls `env.get_losses_since(step - 1)` where the filter is
`timestep >= step - 1` — this re-includes the previous step's losses, which were
already fed on the prior iteration. Pending-loss counts are inflated ~2×,
adaptation triggers early, and the loss analyzer double-counts.

### H5. Curriculum stage budgets are never enforced

`CurriculumStage.episodes` is only used in a log line. Advancement
(`check_curriculum_advance`) is purely `avg_survival >= min_survival_rate`, where
`avg_survival` is a 100-episode window that **spans stages** (early-stage
performance leaks into the next stage's gate). A stage that never reaches its
threshold blocks the curriculum forever regardless of its episode budget.

---

## Medium / minor findings

- **M1. Vestigial machinery presented as active.** `update_return` is never called,
  and `forward_step` feeds constant RTG, zero action embedding, and timestep 0 —
  the Decision Transformer's sequence conditioning is entirely dead; the effective
  policy is `state_embed → 3-token transformer block → heads`. The policy's
  history deques are dead. `PPOTrainer.reward_model` is instantiated and never
  used. `MAMLAdapter.meta_train_step` (the actual MAML outer loop) is never
  called — what runs is plain SGD fine-tuning of a copied head.
  `PrioritizedReplayBuffer.update_priorities` is never called, so priorities are
  frozen at insert values (1.0, or 2.0^α for loss events), `_max_priority` never
  changes, and the importance-sampling weights returned by `sample()` are
  discarded by the only consumer (`_replay_rehearsal`). None of this is
  incorrect per se, but the naming (Decision Transformer, MAML, prioritized
  replay) materially oversells what the algorithm computes.
- **M2. Value loss deviates from PPO2.** `torch.max(v_loss1, v_loss2)` takes the
  max of two batch-mean MSEs rather than the elementwise max before the mean
  (`trainer.py:570-572`). Mild, but it weakens the per-sample clipping rationale.
- **M3. Checkpoint loading is silently lenient.** `load_policy_weights` uses
  `strict=False` (`policy.py:143-146`); an architecture mismatch loads partially
  with no error. `CheckpointManager._checkpoints` only tracks files created in
  the current run, so `keep_last` pruning ignores prior runs' files.
- **M4. Reproducibility gaps.** `PrioritizedReplayBuffer.sample` uses the global
  `np.random` (unseeded); `ThreatSpawner._create_threat`/`_setup_patrol` mix
  Python's `random` with the passed-in numpy generator (works only because
  `env.reset` reseeds the global `random`).
- **M5. Opponent model outputs are near-meaningless.** The LSTM's "confidence" is
  trained toward the constant 0.5 and the third target dim toward 0.0
  (`opponent_model.py:99`); the predicted displacement is a one-step delta
  (~≤1 map unit after ×1000 rescale) yet labeled `time_horizon=5`, and the
  `speed < 5.0 → "hold"` threshold means the action label is almost always
  "hold". Its consumer (recon pre-positioning) therefore steers toward
  ~current threat positions. `train_step` also retrains on every stored window
  of every threat each call — cost grows quadratically with episode length.
- **M6. `resolve_attack` returns `destroyed = damage >= 1.0`**, which is `False`
  for every configured `damage_per_hit` (0.3/0.6); callers correctly ignore it
  and re-check `is_active`, but the return value is a trap.
- **M7. Feature scaling is inconsistent.** `build_observation` normalizes entity
  x/y but leaves velocities (±25) and several raw features unscaled; LayerNorm
  in the entity MLP mitigates this, but it makes the aux-loss slice
  (`threats[:, :, 8:14]`, verified correct against `ThreatState.to_feature_vector`)
  fragile to any feature reordering — worth a named constant or slice derived
  from the model class.
- **M8. Weapon range = sensor range.** `resolve_engagements` passes
  `drone.sensor_range` as `attacker_range`, and hit probability only reaches zero
  at 2× that range — ENGAGE drones can hit from 160 despite an 80 sensor. This
  interacts with the experiment's premise (engage-vs-AA outrange geometry) and
  should be an explicit weapon parameter.

## What checks out

For balance, load-bearing pieces verified sound: the GAE recursion (including
done-reset ordering) is correct; the LR schedule genuinely sets (not multiplies)
LRs with warmup+cosine; the log-ratio clamp and non-finite loss/grad guards do
what they claim; the replay-buffer priority overflow fix is correct; the masked
attention-pooling NaN guard (all-masked rows) is correct; observation building is
a single shared function for positions as claimed; PPO training and inference
share `forward_step` (train/inference parity for the network itself); seeding is
coherent end-to-end; and the experiment harness's methodology (held-out seeds,
constant/random/oracle baselines, probe + value diagnostics) is good practice.
All 21 tests pass.

## Remediation status (addressed on this branch)

The following findings have been fixed in follow-up commits; see the module
docstrings and `tests/` for details.

- **C1 — mission reward aliasing:** `step()` now captures objective-completion
  counts *before* mutation and passes the delta into `_compute_reward`, plus a
  one-off `mission_complete` terminal bonus. Regression tests
  `test_mission_progress_reward_is_not_aliased_to_zero` /
  `test_mission_complete_bonus_paid_once`.
- **C2 — inert PPO factors:** the PPO objective scores only `action_type` by
  default (`ppo_include_inert_factors` restores the old 3-factor objective).
- **C4 — single-context batches + advantage centering:** the training loop no
  longer flushes the rollout at every episode end, so a PPO update can span
  multiple episodes/contexts; `center_advantages=False` keeps the cross-context
  advantage that per-batch mean-centering deletes. The experiment config uses
  `update_interval=512` + `center_advantages=false`.
- **C5 — adaptation/optimizer domain clash:** the learning loop no longer
  overwrites live policy weights during PPO (`apply_weight_updates=False`); the
  counter bank / classifier / EWC / opponent model still learn, consumed
  read-only.
- **H2 — truncation treated as termination:** GAE now bootstraps time-limit
  truncations with V(s_{t+1}) instead of forcing the horizon value to zero
  (`RolloutStep.terminated` / `bootstrap_value`; `test_gae_bootstraps_truncation…`).
- **H4 — double-fed losses:** online adaptation now sees each loss event once.

Added capability (the "divine adaptive engine"): `CounterBank` (outcome-driven,
never-forgetting per-threat counter memory) and `SimpleDomain` (a survival-floor
reflex that shields units from unmastered threats).

Still open (documented, not yet fixed): **H1** (position/priority heads remain
untrainable — would need a stochastic position head in the PPO objective), **H3**
(train/eval `time_scale` parity), **H5** (curriculum stage budgets), and the
medium/minor items **M1–M8**.

## Recommended order of attack

1. Fix the reward: C1 (snapshot states, completion bonus) — this changes the
   objective every other fix is measured against.
2. Clean the PPO signal path: C2 (drop/wire inert factors), C3 (dropout off for
   PPO), H2 (truncation bootstrap), H1 (decide whether position is learnable).
3. Re-run the policy-authority experiment with C4's fix (cross-episode batches,
   rethought advantage normalization) before accepting the "PPO caps at
   constants" conclusion — the current evidence doesn't isolate PPO itself.
4. Then the hygiene items: H3–H5, M1–M8.
