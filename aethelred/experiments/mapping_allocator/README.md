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
