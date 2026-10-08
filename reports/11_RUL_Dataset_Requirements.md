# 11 — RUL Dataset Requirements

## Current decision

The application has **no connected RUL model**. No RUL dataset has been selected,
downloaded or approved in this phase, and no RUL model has been trained.
Dataset approval will be manual before any training. The dashboard displays
“RUL model not yet connected.” and does not fabricate remaining-life or risk
values.

Experiment 2 predicts an observed fault class: Normal / Motor Fault / Inverter
Fault / Battery Fault. Its seven normalized features have unknown original
physical-unit normalization parameters, and it has no timestamps, component
IDs, degradation trajectories or defined end-of-life time. A fault label alone
cannot establish Remaining Useful Life. Existing Experiment 1 `RUL`/`TTF`
columns are also not automatically approved RUL ground truth: their derivation,
component identity, failure definition and observation timing require evidence.
Both experiments and their conclusions remain unchanged.

## Desired target and mandatory approval evidence

Actual RUL is the remaining time, operating hours or cycles until a **defined
component failure/end-of-life threshold**. For a completed run at observation
time/cycle `t`, a documented label may be derived as `end_of_life - t`, in the
same unit. That formula is valid only when identity, ordering, endpoint and
units are established. It is not permission to invent a failure time or turn
unlabeled healthy records into zero-RUL examples.

| Requirement | Evidence needed before approval |
| --- | --- |
| Provenance and relevance | Real or clearly documented experimental data relevant to electric motors, drivetrain or EV components; owner/source, acquisition protocol and whether simulated/synthetic |
| Independent unit identity | Persistent machine/vehicle/component ID, with run ID if a unit has multiple runs; records can be assigned to a genuine degradation history |
| Ordered observations | Timestamps, elapsed operating hours or monotonically increasing cycles; sampling/clock/reset/missing-period behavior is documented |
| Degradation | Repeated observations spanning healthy-to-degraded conditions, not only unrelated fault snapshots |
| Known endpoint | Run-to-failure histories or validated end-of-life thresholds, failure-event record and failure definition; repairs/replacements are identified |
| RUL label protocol | Source labels or a reproducible endpoint-to-observation calculation; label unit, failure threshold, prediction point and any label capping are documented |
| Independent evaluation | Enough independent units/runs for disjoint training, validation and test groups; exact feasibility depends on task and diversity, not an invented minimum sample count |
| Measurement meaning | Sensor definitions, units, calibration/normalization parameters, availability at inference and operating-condition meaning |
| License and access | Clear license, permitted training/redistribution use, attribution, restrictions and stable source/version/checksum |

Complete run-to-failure examples with known endpoints are preferred for the
first supervised regression module. Right-censored units that never reach a
known failure cannot be assigned exact RUL by assuming the last observation
was failure. If retained, censoring needs an explicit method and an appropriate
evaluation design, reviewed separately. Units with undocumented endpoint
status must not silently be used as exact labels.

## Useful sensor and operating context

Seek as many relevant measurements as practical, without requiring every
dataset to contain every sensor:

- Load, speed, duty cycle and operating/environmental conditions.
- Vibration, temperature, current and voltage where available.
- Torque/RPM where meaningful for motors or drivetrains.
- Battery-specific degradation information for a battery RUL module.
- Component version/type, maintenance/repair history and measurement quality
  where documented and available at the prediction point.

Features must be measured before or at inference. Future failure annotations
may create training labels, but must not become model inputs. “Health scores”
or degradation indices require a derivation/timing audit, not blind inclusion.
Physical readings must retain known units; any normalization must be
reproducible and fitted only within the training data.

## Validation and leakage rules

1. Audit schema, ranges, units, missing/duplicate observations, ordering,
   endpoint validity, failure modes and censoring by component/run.
2. Assign independent machines/runs to train/validation/test before creating
   overlapping windows. Keep related units or repeated runs together where
   their shared information would leak. Randomly splitting rows from one run
   across partitions is not an independent generalization test.
3. Use time-aware validation for forecasting claims and disjoint-unit
   validation for cross-machine claims. Avoid implying both from only one.
4. Build windows with history available at or before each observation; do not
   use future readings, centered smoothing or test-derived statistics.
5. Fit imputers, scalers, feature selection and other learned transformations
   on the appropriate training fold only. Final test data stays untouched
   until the model/setup has been selected.
6. Check that endpoint-derived labels decrease consistently within an
   unrepaired completed run, remain nonnegative and use the declared units.
   Define how reset/repair/replacement starts a new lifecycle.
7. Report suitable RUL errors (for example MAE/RMSE in declared units),
   performance by unit/operating condition and over/under-estimation behavior.
   Define uncertainty/calibration and decision costs before any risk method.
   Do not invent a risk score from fault confidence.

Failure-event fields can legitimately be available retrospectively for
training labels. They are not evidence that the endpoint is available at
deployment. Document this distinction explicitly in the approved dataset.

## Manual approval checklist and next step

Before downloading/using a candidate, provide its source, license, component
relevance, schema/sample description, number and variety of independent runs,
sampling/units, endpoint/label protocol, censoring status, sensor timing and
proposed group/time-safe split. Record known coverage limits and unresolved
questions. A candidate lacking identity, ordered degradation or a credible
failure endpoint is not ready for the first supervised RUL module.

After user approval: obtain the original files, preserve raw data, audit them,
approve preprocessing/label generation and only then request RUL training
approval. No internet dataset is automatically downloaded or accepted here.

Reserved locations: `data/rul/` (local approved data), `src/rul/` (contracts and
future adapters), `models/rul/` (future fitted artifacts), `reports/rul/`
(future evidence). `RULPrediction` requires component, machine ID, a real
estimated RUL, unit and model version; there is no numerical default. Each
specialized model will own its features, preprocessing, units and validation.
