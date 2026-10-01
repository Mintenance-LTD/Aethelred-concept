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

The next research step is a larger, predefined independent layout suite and
comparison with a planner optimizing remaining mission workload. Any learned
candidate should demonstrate a useful time/distance tradeoff before deployment
approval is considered.
