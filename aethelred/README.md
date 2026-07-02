# Aethelred — Adaptive Autonomous Swarm Intelligence (Simulation)

A **research simulation** of the "Aethelred" concept: a mother-drone tactical AI
that commands an expendable swarm and adapts to threats via a continual-learning
loop. This repository is a 2D Gymnasium environment plus a PyTorch policy stack —
**simulation only, with no hardware, actuation, or networking**.

> See `../aethelred-concept.html` for the concept document. As that document
> states (§7), any real-world system of this kind would require robust
> human-in-the-loop controls and clear rules of engagement. This codebase
> implements none of that and is intended for research/educational use in
> simulation.

## Architecture

```
core/         Pydantic state models, enums, actions, events, interfaces
simulation/   Gymnasium env, battlefield/terrain, physics, entities, threat AI
tactical_ai/  State encoder (attention pooling) + Decision Transformer + action head
learning/     PPO trainer (GAE, clipping, curriculum) + the 5-phase learning loop
adaptation/   MAML + EWC + prioritized replay + threat classification ("Mahoraga")
swarm/        Coordinator, mother drone, lightweight units, comms, distillation
deployment/   Safety manager (geofence/watchdog/RTL) + ONNX/TorchScript export
```

## Install

```bash
pip install -e ".[dev]"      # from the aethelred/ directory
```

## Run

```bash
# Watch a simulation with a random policy
python scripts/run_simulation.py --steps 200 --seed 42

# Train the tactical AI with PPO (this actually updates the policy now)
python scripts/train.py --mode online --episodes 100
python scripts/train.py --mode curriculum --episodes 200
python scripts/train.py --resume checkpoints/best_policy.pt

# Run the 5-phase learning/adaptation loop demonstration
python scripts/run_learning_loop.py --episodes 5

# Stress-test scenarios + safety + decision latency
python scripts/stress_test.py --all

# Export a trained model for edge inference
python scripts/export_model.py --checkpoint checkpoints/best_policy.pt
```

TensorBoard logs are written to `runs/`; checkpoints to `checkpoints/`.

## Testing

```bash
pytest -q          # unit tests
ruff check .       # lint
```

## Reproducibility

`aethelred.utils.seeding.set_seed` seeds Python, NumPy, and PyTorch together, and
the environment reseeds Python's `random` on `reset(seed=...)`. With a fixed seed
and fixed actions the simulation is deterministic.

## Status / recent fixes

The training and adaptation pipelines were previously non-functional. Fixed:

- **PPO now trains the policy.** The update re-encodes stored observations and
  re-runs the transformer, so gradients reach the encoder + transformer (not just
  the value head). The training scripts actually call the update loop.
- **LR schedule** no longer collapses to zero (proper warmup + cosine decay).
- **Adaptation** learns a real threat→counter mapping (not a constant), and the
  replay buffer and EWC are actually exercised for continual learning.
- **Config** hyperparameters (`tactical_ai`, `state_encoder`) now drive model
  construction instead of being ignored; the loader warns on unknown keys.
- **Sensor-noise injection** no longer mutates ground-truth state.
- **Heterogeneous swarm:** the env decomposes one high-level command into per-role
  actions — ENGAGE units prosecute threats (distributed), RECON observe/evade, EW/RELAY
  hold the mesh — instead of broadcasting one action to every drone.
- **Threat mastery wired:** neutralizations now feed the threat classifier, so
  counter-effectiveness / mastery metrics populate during runs.
- Reproducible seeding; corrected exchange-ratio threat metrics; curriculum stages no
  longer leak settings; lint clean; test suite covers training, sim, adaptation,
  swarm, and safety.

- **Comms-aware autonomy:** units beyond comm range or under EW jamming fall back to
  local autonomous decisions (`SwarmUnit`), instead of the central command.
- **Predictions consumed:** opponent-model predictions now pre-position recon units
  toward anticipated threats.
- **Single observation builder** (`state_encoder.build_observation`) shared by the env
  and policy, removing the position-normalization drift.
- `PolicyDistiller` is now a working, tested utility; `ActionValidator` enforces the
  configured commanded-speed cap.

- **Train/inference parity:** both PPO training and `policy.decide` now go through a
  single `policy.forward_step`, so the deployed policy is exactly the one optimized.

### Training results (demonstration)

PPO does learn to improve survival when the objective rewards it. On a fixed scenario
(seed 5, aggressive AA threats) survival improved from **~50% to ~71%** (deterministic
eval) using a survival-focused reward — see `configs/survival_train.yaml`:

```bash
python scripts/train.py --config configs/survival_train.yaml --episodes 120 --no-noise --seed 5
```

Notes learned along the way: the default *composite* reward (survival + objectives +
kills − losses) is not the same as survival, so naive training won't raise survival;
a sharp `loss_penalty` plus strong `entropy_coef` (exploration) is needed for PPO to
discover the evade/withdraw behavior. Checkpoints are selected by single-episode reward,
which is noisy — the latest checkpoint is often a better survival policy than `best_policy.pt`.

### Divine adaptive engine (post-audit improvements)

Following the algorithm audit (`../ALGORITHM_AUDIT.md`), the adaptation path was
reworked so it stops fighting the PPO optimizer and starts learning real counters:

- **Counter bank (`adaptation/counter_bank.py`):** a per-threat contextual bandit
  over the eight high-level actions. It learns which action actually counters each
  threat *from observed outcomes*, only trusts a counter once the "wheel has
  turned" enough times, and — being a table — never catastrophically forgets one.
  Exposed via `AdaptationEngine.recommend_counter` / `mastery_of` / `record_outcome`.
- **No more domain clash (audit C5):** during PPO training the learning loop no
  longer overwrites the live policy weights mid-rollout (`apply_weight_updates`
  is forced off in `scripts/train.py`); the optimizer is the sole writer. The
  standalone learning-loop demo keeps the weight write for illustration.
- **Simple Domain survival floor (`deployment/safety.py::SimpleDomain`):** an
  opt-in reflex (`--simple-domain`) that pushes any unit sitting inside the kill
  range of an *unmastered* threat back out (EVADE), so exploration against a
  lethal threat can't collect a free kill on our units. Mastery comes from the
  counter bank, so the shield relaxes automatically as the engine learns.
- **Cleaner PPO signal (audit C2):** the PPO objective scores only the action
  factor the environment consumes (`action_type`) by default; the inert
  `formation`/`target_index` factors are excluded unless
  `ppo_include_inert_factors` is set.
- **Double-fed losses fixed (audit H4):** online adaptation now sees each loss
  event once.

### Known remaining work (design notes, not bugs)

- `ModelExporter` uses `torch.jit.trace`, which bakes in trace-time control flow;
  revisit if exporting models with heavily data-dependent branching.
- Checkpoint selection is by single-episode reward; an eval-based or running-average
  criterion would pick more reliable policies.
