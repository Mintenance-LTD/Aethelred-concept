# SIL qualification handoff

This repository's local simulator is useful for regression and safety-boundary
testing. It is not target-autopilot software-in-loop (SIL) evidence and must
not be presented as such.

## Required external inputs

Before SIL execution, the programme owner must select and provide:

- target autopilot and version;
- SIL launcher, transport endpoint, and vehicle model;
- target edge/runtime platform details;
- the flight-controller geofence and failsafe configuration;
- non-offensive mission fixtures and a designated operator identity;
- a secret-store supplied attestation key or an external attestation verifier.

No target autopilot, MAVLink transport, or SIL launcher is configured in this
repository as of this handoff.

## Preconditions for every run

1. Start the target autopilot SIL with its own geofence and failure response
   enabled. Aethelred supplements these safeguards; it does not replace them.
2. Activate an attested release whose manifest matches the runtime code,
   configuration, observation schema, data reference, environment, and build
   provenance.
3. Construct the operational policy only through `ApprovedIntentPolicy.load`.
   The policy may propose typed intents only; the signed-intent, safety,
   lifecycle, health, command-arbiter, and adapter gates remain mandatory.
4. Configure health reports for estimator, adapter, safety supervisor, and
   policy process. Missing, stale, or unhealthy reports must remove authority.

## Minimum evidence per scenario

Record a hash-bound scenario result and the corresponding audit journal for:

- survey, inspection, mapping, relay, and disaster-search missions;
- degraded GPS, communications blackout/reconnection, and adverse weather;
- stale, frozen, biased, corrupt, and lost sensor observations;
- estimator, policy, adapter, and communications-process failures;
- deadline misses, duplicated acknowledgements, stale commands, and restart
  recovery.

Each result must show that no vehicle command bypassed the safety and command
arbiter path, and include mission revision, runtime identity, model digest,
configuration digest, policy proposal, safety decision, command sequence, and
vehicle acknowledgement.

## Promotion boundary

SIL evidence must be declared as held-out scenario categories in the release
evaluation and enforced by `PromotionPolicy.required_scenario_categories`.
Passing a local simulator test suite alone is insufficient to activate a target
vehicle release.
