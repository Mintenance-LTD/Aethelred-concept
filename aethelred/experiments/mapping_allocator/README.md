# Offline mapping allocator experiment

This experiment asks whether learning task duration improves bounded civilian
mapping allocation. It uses a small ridge regressor, not the tactical policy.
The coordinator validates proposals; authenticated intents and deterministic
safety authorisation remain mandatory for every simulated vehicle movement.

## Reference run

Training: 2,560 synthetic duration examples from seeds 0–63. Validation: 640
examples from seeds 500–515. Validation mean absolute error: 0.382 ticks.
The model is trained once with fixed regularisation; evaluation never changes
its weights. The target is the documented discrete movement model's task time,
not a label derived from held-out mission outcomes.

The held-out comparison uses layout seeds 9100–9103, each with nominal operation,
unit loss, communications loss, stale sensors, and coordinator restart. Initial
geometry, unit speeds, and fault schedules are matched across all three planners.

| Allocator | Completed missions | Mean ticks | Mean distance | Duplicate samples | Observed boundary violations |
| --- | --- | --- | --- | --- | --- |
| Nearest task | 20/20 | 29.60 | 327.56 | 0 | 0 |
| Analytic batch balancing | 20/20 | 30.95 | 339.37 | 0 | 0 |
| Learned batch balancing | 20/20 | 29.40 | 329.16 | 0 | 0 |

The learned candidate clears the predefined local mean-time comparison. Its
advantage over nearest-task allocation is only 0.68%, with 0.49% more travel.
Individual cases regress: on seed 9102 with unit loss, nearest takes 40 ticks,
learned balancing 43, and analytic balancing 52. The batch objective is myopic;
an accurate prediction of the next task's duration does not necessarily minimize
the complete mission's duration.

Four layout seeds are four independent layouts; their five scenario variants
are correlated. These descriptive results do not demonstrate statistical
significance, field performance, or readiness to replace the default planner.
Nearest-task allocation remains the default. No model was registered, promoted,
or activated as a production release.

`REFERENCE_RESULTS.json` preserves model coefficients and split provenance,
source/environment hashes, all 60 mission measurements, and each layout. The
full local report and replay journals live under
`.artifacts/mapping-benchmark-heldout-20261001/` and are excluded from Git.
The source hashes identify the working tree used for measurement. A subsequent
producer correction hashes persisted report bytes rather than the pre-write
string, accounting for Windows newline translation; measured rows and weights
were unchanged.

## Reproduce

From `aethelred/`, choose an empty output directory:

```bash
python scripts/benchmark_mapping_allocators.py --output .artifacts/mapping-comparison
```

Open `report.html` for the summary and each mission's `replay.html` for its
assignment, safety, and command evidence. The report checksum binds the actual
persisted JSON bytes. Evaluation seeds overlapping training or validation are
rejected before artifact creation.

## Independent frozen-model evaluation

The predefined protocol in `INDEPENDENT_EVALUATION.md` was completed on 16 new
layout seeds (10000–10015), with all five scenarios and all three planners:
240 missions total. The reference model's exact bytes and coefficients were
unchanged. No tuning or retraining used these outcomes. All missions completed
with zero duplicate observations and zero observed command-boundary violations.

| Allocator | Completed | Mean ticks | Mean distance |
| --- | --- | --- | --- |
| Nearest task | 80/80 | 23.8625 | 328.25555 |
| Analytic batch balancing | 80/80 | 22.9125 | 325.50280 |
| Learned batch balancing | 80/80 | 24.2875 | 330.95500 |

The earlier small learned advantage did **not** generalize to this cohort.
Learned balancing took 1.78% longer than nearest-task, with 0.82% more travel.
Its layout-averaged time was better on four layouts, worse on seven, and tied
on five. Analytic balancing had the best aggregate time and distance here,
but this alone does not justify changing the production default.

Paired learned-minus-baseline differences, with 95% percentile bootstrap
intervals resampling complete layout clusters (not individual scenarios):

| Baseline | Mean time difference | Time interval | Mean distance difference | Distance interval |
| --- | --- | --- | --- | --- |
| Nearest task | +0.425 ticks | +0.025 to +0.9375 | +2.69945 | -1.81571 to +6.40263 |
| Analytic balancing | +1.375 ticks | +0.375 to +2.525 | +5.45220 | -1.47347 to +12.78428 |

These are exploratory, unadjusted intervals from a small synthetic cohort;
they do not establish field performance. Communications-loss cases accounted
for the largest average time regression versus nearest-task (+1.125 ticks).
Stale-sensor cases averaged 0.5 ticks faster. Complete scenario breakdowns and
per-layout differences are retained, rather than reporting only favorable cases.

The candidate fails the existing descriptive local mean-time comparison.
Nearest-task remains the default. No model was promoted or deployed.
`INDEPENDENT_RESULTS.json` preserves split/model/source provenance, the analysis,
all 16 layouts, and all 240 measurements. Each compact measurement row follows
the explicit `measurement_columns` schema. The report checksum binds the full
local JSON under `.artifacts/mapping-benchmark-independent-20261001/`, where
readable reports, individual measurements, replays, and verified journals remain.

The next research step is a planner that considers remaining mission workload
and travel, compared with the existing baselines. This cohort is now inspected
research evidence: any subsequent tuned candidate needs a fresh, predefined
held-out cohort. Keep local simulation qualification separate from eventual
autopilot and field qualification.

## Remaining-work planner

The deterministic, simulation-only `mission-workload/v1` candidate plans the
whole currently available remaining workload. It accounts for busy units' locked
task prefixes and retains completed samples. Future routes are forecasts, not
ownership reservations; only the next task for each idle unit is proposed.
The coordinator still validates every batch and controls all leases. Every
vehicle movement retains the authenticated intent and safety authorization path.

It minimizes projected completion time under observed constant speeds, using
subset route dynamic programming and exhaustive fleet partitions. Distance
breaks ties between retained minimum-time routes; it is not a global travel
optimizer. Bounds are three units, twelve unfinished cells, and four samples per
cell. Independent small-case exhaustive-oracle tests check the time objective,
and mission tests cover all five scenarios, partial progress, restart, and
rejection of invalid contextual proposals.

The development comparison (four layouts, two scenarios, four planners) completed
all 32 missions. The candidate was then frozen before evaluating sixteen new
layouts (11000–11015) with all five scenarios. All 320 final missions completed
with zero duplicate samples and zero observed command-boundary violations.
The learned comparator's artifact bytes and coefficients were unchanged.

| Planner | Completed | Mean ticks | Mean distance | Maximum proposal wall time |
| --- | --- | --- | --- | --- |
| Nearest task | 80/80 | 26.0750 | 382.12334 | 0.00012 s |
| Analytic balancing | 80/80 | 25.7625 | 389.10066 | 0.00961 s |
| Frozen learned balancing | 80/80 | 25.5000 | 381.85370 | 0.00887 s |
| Remaining workload | 80/80 | 23.0125 | 349.59328 | 1.79917 s |

Against nearest-task, workload planning used 11.74% fewer simulated ticks and
8.51% less travel. Layout-averaged time and distance improved on all sixteen
layouts. Individual scenario regressions remain: seed 11011 communications loss
took 22 ticks versus nearest's 21, and stale sensors took 21 versus nearest's 20.
Against the learned comparator, one layout used slightly more travel.

Paired workload-minus-baseline layout-cluster bootstrap intervals:

| Baseline | Mean ticks difference | 95% time interval | Mean travel difference | 95% travel interval |
| --- | --- | --- | --- | --- |
| Nearest task | -3.0625 | -3.7500 to -2.3750 | -32.53006 | -42.48230 to -23.24211 |
| Analytic balancing | -2.7500 | -3.5875 to -2.0125 | -39.50739 | -53.79558 to -26.45555 |
| Frozen learned balancing | -2.4875 | -3.1750 to -1.8875 | -32.26043 | -43.19760 to -21.98631 |

These exploratory, unadjusted intervals describe this synthetic generator only.
The fixed fault schedule targets mapper-1 at prescribed ticks, not every unit
or every operational phase. On workload seed 11006, mapper-1 had no active
assignment at communications loss, so no stale-token rejection was exercised.
That case is retained, not discarded; rotated active-unit fault testing remains
necessary before stronger recovery claims.

There is a substantial compute tradeoff: workload proposals totaled 0.457 seconds
per mission on average and peaked at 1.799 seconds per call. These wall times
include concurrent-process contention, and are not deadline guarantees. The
candidate clears the descriptive local mean-time comparison, but remains opt-in
and simulation-only. Nearest-task is still the default; no production planner
or model was promoted, registered, or activated.

`WORKLOAD_EVALUATION.md` contains the predefined protocol and exact run commands.
`WORKLOAD_RESULTS.json` preserves source/split/model provenance, all sixteen
layouts, all 320 measurements including planning times, and the paired analysis.
Each compact row follows its explicit `measurement_columns` schema. Full local
reports, measurements, replays, and verified journals remain under
`.artifacts/mapping-workload-heldout-20261001/`; the development evidence is under
`.artifacts/mapping-workload-development-20261001/`.

Next: deadline-bounded planning with a verified deterministic fallback, followed
by rotated active-unit fault qualification. Battery-aware workload planning and
return-to-base constraints also remain open. These evaluated layouts are now
inspected research evidence; use a fresh predefined cohort for future tuned
candidates. Keep all qualification in local simulation until a flight stack is
explicitly selected.
