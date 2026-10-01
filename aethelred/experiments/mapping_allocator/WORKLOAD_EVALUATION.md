# Remaining-work planner: fixed evaluation protocol

This civilian mapping experiment follows the frozen-model independent comparison.
Nearest-task remains the default; no flight stack or production release changes.

## Planner and development

`mission-workload/v1` computes a minimum-duration route for every pending subset
and observed unit using Held-Karp dynamic programming. It then enumerates fleet
partitions to minimize projected completion time. Distance breaks ties between
the retained minimum-time routes; this is not global distance/Pareto optimization.
The search is bounded to three units, twelve unfinished cells, and four samples
per cell. It is deterministic, stateless, and uses no learned weights.

Current online eligible busy units are included with their active task locked
as a prefix. Partial progress is retained. Only first tasks for idle units are
proposed; future routes grant no leases or reservations. An idle unit can wait
when routing future work to a faster busy unit gives a better forecast.
Unknown future faults and recovery times are not visible to the planner.
Work owned by a missing unit remains unavailable until its lease expires;
that uncontrollable work is not included in the observed-fleet time forecast.

The optional read-only coordinator context does not bypass proposal validation,
authentication, deterministic safety authorization, or stale-owner fencing.
Legacy allocators retain their two-argument interface and behavior. Context and
planner source fingerprints are recorded for auditability.

Development uses constructed oracle cases, the already-inspected research
layouts, and seeds 10500–10503. A twelve-cell planning-size check on inspected
seed 10011 took approximately 1.8 seconds on this machine. This is not a hard
latency guarantee. No outcome from the final cohort is used to tune the planner.

## Final cohort and analysis

- Seeds 11000–11015 inclusive: sixteen new independent generated layouts.
- Five scenarios: nominal, unit loss, communications loss, stale sensors,
  and coordinator restart. Variants within one layout are correlated.
- Four planners: nearest-task, analytic-balanced, frozen learned-balanced,
  and mission-workload. Total: 320 missions, all retained without optional stopping.
- Learned comparator artifact is unchanged:
  SHA256 `79f7e19dc18cc7db97c9516ac8a922ab46c65f929525311777efa7f556d29533`.
- Exclude training seeds 0–63, validation seeds 500–515, observed layouts
  9100–9103 and 10000–10015, and development seeds 10500–10503.

Compare workload minus each of the three baselines for mission ticks and travel.
Average scenario differences within layouts; resample whole layout clusters
20,000 times using random seed 20261001 for exploratory percentile 95% intervals.
Report per-scenario means and better/worse/tied layouts. Intervals are unadjusted
for multiple comparisons and apply to this abstract generator, not field flight.

Retain completion, coverage, duplicate samples, and observed command-boundary
violations. Incomplete missions remain in the cohort; capped ticks are not an
estimate of eventual completion time. Report wall time for planner proposal
calls separately from simulated mission time, including process contention.
Runtime deadlines, battery-aware mission planning, camera coverage, autopilot
integration, and flight dynamics are not established by this experiment.

The local mean-time comparison is descriptive, not a deployment gate.
Nearest-task remains the default regardless of outcomes. Do not promote a
planner automatically or modify it after seeing this cohort's measurements.

## Reproduce

From `aethelred/`, choose an empty output directory. PowerShell:

```powershell
.venv/Scripts/python.exe scripts/benchmark_mapping_allocators.py `
  --output .artifacts/mapping-workload-heldout-20261001 `
  --model .artifacts/mapping-benchmark-heldout-20261001/duration-model.json `
  --expected-sha256 79f7e19dc18cc7db97c9516ac8a922ab46c65f929525311777efa7f556d29533 `
  --include-workload --seeds (11000..11015) --workers 4 `
  --excluded-seeds ((9100..9103) + (10000..10015) + (10500..10503))
```

The evaluation plan and source/model fingerprints are saved before missions.
Each mission retains its measurement, replay, and verified audit journals;
the persisted full report is checksum-bound. Interrupted runs preserve evidence
but require a fresh output directory for rerunning the complete comparison.

For one mission, no trained model is needed:

```powershell
.venv/Scripts/python.exe scripts/run_mapping_mission.py `
  --allocator workload --scenario comms_loss --layout-seed 10500 `
  --output .artifacts/workload-demo
```
