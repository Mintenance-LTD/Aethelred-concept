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
runtime/      Non-offensive mission, intent, safety-authorisation, and command contracts
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

## Civilian mapping mission and recovery

The deterministic mapping demonstration connects three units through the
authenticated operational command path. A coordinator partitions a 60 by 40
metre area into six cells with four sample locations each. Versioned assignment
leases, acknowledgements, heartbeats, cancellation, and durable progress govern
ownership. Units visit samples through safety-authorised commands; unfinished
work is reallocated when an assignment expires.

```bash
python scripts/run_mapping_mission.py --scenario unit_loss --output .artifacts/mapping-unit-loss
python scripts/run_mapping_mission.py --scenario comms_loss --output .artifacts/mapping-comms-loss
python scripts/run_mapping_mission.py --scenario stale_sensor --output .artifacts/mapping-stale-sensor
python scripts/run_mapping_mission.py --scenario coordinator_restart --output .artifacts/mapping-restart
```

Use `--scenario nominal` for the baseline. Each output directory must be empty.
Open `replay.html` for a readable event timeline. `summary.json` reports sample
completion, elapsed ticks, duplicate visits, travel distance, expired leases,
rejected intents, and reassignment delay. Verified source journals and the full
`replay.json` retain assignment, authentication, telemetry, safety, movement,
command sequence, acknowledgement, and sample evidence.

Mapping coverage here means visitation of specified sample points. This is an
abstract 2-D movement and energy model, with a deterministic planner and local
in-process communication. Camera coverage, flight dynamics, networking, learned
allocation, and target-autopilot SIL remain separate qualification work. The
task journal has one active coordinator writer; coordinator restart recovers
existing leases and progress without extending authority. A full vehicle runtime
restart still requires the lifecycle's explicit safe-state recovery procedure.

Operational positions are copied into immutable values independent of tactical
types. Configuration and release transitions expose new state only after their
audit writes succeed. Model integrations must now provide a trusted attestation
verifier to `ActiveReleaseVerifier(ledger, verifier)`; loading rechecks promotion
requirements and attestation validity, including after journal recovery.

## Offline allocator comparison

The mapping coordinator accepts allocation proposals through `MappingAllocator`.
It validates the entire batch before issuing task leases: vehicles must be idle,
available, assigned to the mission, and healthy enough; tasks must be pending;
no vehicle or task may appear twice. A proposed allocation does not authorise a
vehicle command. Every resulting movement still uses the authenticated safety
and command path.

```bash
python scripts/benchmark_mapping_allocators.py --output .artifacts/mapping-benchmark
```

This trains a four-coefficient ridge model offline to predict task duration from
travel distance, speed, remaining route length, and sample count. Training uses
seeds 0–63, validation uses 500–515, and the default mission comparison uses
9100–9103. Overlapping splits are rejected before training. The candidate weights
are immutable after loading; the coordinator has no training interface.

Three allocators run matched layouts and fault schedules: nearest task, analytic
batch balancing, and learned batch balancing. The analytic and learned variants
search the same assignment combinations, isolating the cost model from the
search method. The default suite runs 20 missions per allocator across nominal,
unit loss, communications loss, stale sensors, and coordinator restart.

`report.html` presents the comparison; `report.json` records individual outcomes,
source hashes, environment, split provenance, model digest, and observed command
boundary violations. Each mission retains its replay and verified journals.
These synthetic results are descriptive and never activate a production release.
Scenarios sharing a layout seed are correlated; 20 mission runs do not represent
20 independent layouts or establish statistical significance.
The learned candidate must improve mean completion ticks against both controls
with full completion, zero duplicate visits, and zero observed command-boundary
violations to clear the local comparison. Clearing that comparison alone does
not establish operational readiness or approval.

Run an individual experimental candidate with:

```bash
python scripts/run_mapping_mission.py --scenario comms_loss --layout-seed 9200 --allocator learned --model .artifacts/mapping-benchmark/duration-model.json --output .artifacts/learned-mapping
```

The recorded [reference experiment](experiments/mapping_allocator/README.md)
includes the measured results and their limits. The nearest-task planner remains
the default; the learned allocator is an explicit local research option.

## Testing

```bash
pytest -q          # unit tests
ruff check .       # lint
```

## Reproducibility

`aethelred.utils.seeding.set_seed` seeds Python, NumPy, and PyTorch together, and
the environment reseeds Python's `random` on `reset(seed=...)`. With a fixed seed
and fixed actions the simulation is deterministic.

## Safety execution boundary

The training path decodes a proposed Gym action once, sends the resulting
`TacticalDecision` through `SafetyExecutionGateway`, and executes only the
returned `AuthorisedDecision` via `AethelredEnv.step_decision()`. This ensures
that emergency-stop, return-to-launch, geofence, and action-validator changes
apply to the exact decision used by the simulator rather than to a separately
decoded action.

Runtime YAML is loaded strictly: unknown keys, malformed nested sections,
invalid core values, and conflicting root/training device settings fail with a
`ConfigurationError`.

## Operational runtime foundation

`aethelred.runtime` is a domain-neutral, non-offensive control boundary for
future bounded uses such as survey, inspection, mapping, relay, and search.
`IntentProposal` objects have no direct execution authority. An
`OperationalSafetySupervisor` validates mission identity, vehicle assignment,
capability allow-lists, state revision/freshness, expiry, and vehicle health;
only its `AuthorisedCommand` can pass through `CommandArbiter` to an adapter.
The simulator-only `SimulatorCommandAdapter` lives in
`aethelred.simulation.operational_adapter`; it uses the simulator's
decision-only execution path and maps every allowed operational capability to a
non-offensive simulator action. It is deliberately excluded from the production
runtime package because it depends on the simulator's tactical representation.

For a production-facing entry point, `AuthenticatedOperationalControlLoop`
requires an `AuthenticatedIntent` validated by `IntentAuthenticator` before the
proposal reaches the safety supervisor. The envelope is HMAC-authenticated,
short-lived, and replay-resistant; it is still only an intent and never bypasses
mission or safety authorisation.

## Offline adaptation boundary

`LearningLoop` may derive an `OfflineAdaptationCandidate` from simulated loss
data, but it never loads those candidate weights into the active policy or
propagates them to swarm units. Candidates include a digest of their base
weights and must pass independent evaluation, human approval, signing, and the
deployment manifest process before any future use.

`aethelred.deployment.ModelPromotionGate` now enforces this evidence boundary:
a candidate needs held-out scenario coverage, improvement over its declared
baseline on required metrics, recorded passing safety checks, a matching model
manifest/report hash, and a named human approval. It creates an approval record
only—it does not load a model or dispatch any command.

`HeldOutEvaluator` runs the same declared scenario set for a candidate and its
baseline, aggregates comparable metrics and safety outcomes, and writes a
canonical evaluation report. The report's SHA-256 is calculated from its exact
persisted bytes and is therefore directly usable by `ModelManifest` and the
promotion gate.

`ReleaseLedger` records approved release registration, activation, and rollback
events to the durable JSONL audit journal. A rollback can target only a
previously approved release and requires a named operator plus rationale; it
updates release governance state only, never an active runtime model.
On startup the ledger replays and validates the journal, restoring the active
release and failing closed if release identifiers or lifecycle history conflict.
Every audit event is hash-chained to its predecessor, so replay also fails
closed when a persisted event is altered, deleted from the middle of the log, or
otherwise breaks the recorded sequence.

At the final execution boundary, each authorised command is short-lived and
single-use. The arbiter rejects expired or replayed command IDs before calling
an adapter, records dispatch before invoking that adapter, reconstructs consumed
IDs from the verified journal after restart, and rejects acknowledgements that
do not identify the same command.

Each `Mission` also carries finite, closed `OperatingArea` bounds. The runtime
authoriser rejects a proposal when its current vehicle position or requested
target lies outside that approved area, independently of simulator geofencing.

For the authenticated operational loop, each mission also names its authorised
intent issuers. A valid HMAC alone is insufficient: a signed proposal from an
issuer outside the mission allowlist is journalled and rejected before safety
authorisation or execution.

`OperationalControlLoop.submit` rejects raw policy proposals. Production callers
must use `AuthenticatedOperationalControlLoop`, which verifies the signed envelope
and issuer binding before the loop records a safety decision or reaches an adapter.

`WorldState` includes battery reserve, localisation quality, sensor timestamp,
communications health, operator-link status, and runtime health. The safety
supervisor rejects stale, non-finite, unhealthy, low-localisation, or unsafe-link
state; low battery and lost links permit only `HOLD` or `RETURN_HOME` intents.
Every verified proposal writes a hash-chained `telemetry_observed` record before
its intent and safety-decision records, preserving the exact operational snapshot
considered by the authoriser.

`MissionRegistry` persists operator-accountable mission registrations and their
strictly increasing revisions in the same hash-chained journal. The authenticated
loop rejects unknown or superseded mission objects before it verifies an envelope
or reaches the safety supervisor.

`ActiveReleaseVerifier` is the runtime artefact boundary: it requires an active
ledger registration and compares the exact model digest, filename, code revision,
canonical runtime configuration, observation schema, and runtime target before it
will call a supplied model loader.

## Status / recent fixes

The training and adaptation pipelines were previously non-functional. Fixed:

- **PPO now trains the policy.** The update re-encodes stored observations and
  re-runs the transformer, so gradients reach the encoder + transformer (not just
  the value head). The training scripts actually call the update loop.
- **LR schedule** no longer collapses to zero (proper warmup + cosine decay).
- **Adaptation** learns a real threat→counter mapping (not a constant), and the
  replay buffer and EWC are actually exercised for continual learning.
- **Config** hyperparameters (`tactical_ai`, `state_encoder`) now drive model
  construction; runtime loading rejects unknown or malformed configuration.
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
discover the evade/withdraw behavior. The trainer selects `best_policy.pt` by
rolling reward over its most recent 100 episodes; release promotion separately
requires held-out evidence.

### Remaining deployment-context work

- Hardware-in-the-loop validation, actuator protocol integration, and credential
  management need an authorized target platform and operational environment; they
  are deliberately not simulated as real-world deployment claims in this repository.
