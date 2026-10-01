# Frozen-model independent evaluation

This protocol is fixed before collecting the expanded mission outcomes.
It is a local civilian mapping simulation, not autopilot or flight qualification.

## Cohort and model

- Layout seeds: 10000 through 10015 inclusive (16 independent generated layouts).
- All five scenarios: nominal, unit loss, communications loss, stale sensors,
  and coordinator restart. Scenario variants of one layout are correlated.
- Three planners: nearest-task, analytic-balanced, learned-balanced.
- Total: 240 matched missions; no optional stopping or removal of poor outcomes.
- The existing reference artifact is reused without fitting or weight updates.
  SHA256: `79f7e19dc18cc7db97c9516ac8a922ab46c65f929525311777efa7f556d29533`.
- Training seeds 0–63, validation seeds 500–515, and previously evaluated layout
  seeds 9100–9103 are excluded. Seeds are a deterministic cohort from the existing
  generator, not a claim of real-world distribution coverage.

## Measurements and interpretation

Retain the full authentication, safety, command, and assignment audit paths.
Independent processes write separate mission evidence directories; concurrency
does not change simulated time, fault schedules, or planner behavior.

Report completion, duplicate observations, observed command-boundary violations,
mission ticks, and distance for every case. For each baseline compute learned
minus baseline differences, first averaging all five scenarios within each
layout. Negative differences favor the learned planner. Count better, worse,
and tied layouts, and report a separate mean difference for each scenario.

Use a layout-cluster percentile bootstrap with 20,000 resamples and fixed random
seed 20261001 to estimate 95% intervals for paired time and distance differences.
Do not count scenario variants as independent observations. These intervals are
exploratory, unadjusted for multiple comparisons, and describe this generator
only. One layout is insufficient for an interval. Incomplete missions remain
in the report; capped time is not an estimate of eventual completion time.

The existing local mean-time comparison is descriptive, not a release gate.
Nearest-task remains the default regardless of this run; no automatic promotion
or production approval is authorized by benchmark results.

## Run

From `aethelred/`, after generating the reference artifact, choose an empty output
directory. In PowerShell:

```powershell
.venv/Scripts/python.exe scripts/benchmark_mapping_allocators.py `
  --output .artifacts/mapping-benchmark-independent-20261001 `
  --model .artifacts/mapping-benchmark-heldout-20261001/duration-model.json `
  --expected-sha256 79f7e19dc18cc7db97c9516ac8a922ab46c65f929525311777efa7f556d29533 `
  --seeds (10000..10015) --excluded-seeds 9100 9101 9102 9103 --workers 4
```

The model bytes are copied unchanged into the output. The evaluation plan and
source fingerprints are saved before missions run. Each completed mission has
its own measurement and replay. Interrupted runs retain evidence but are not
resumable; use a new empty output directory for a complete rerun. A changed model
or source during evaluation prevents the final comparison report.
