# Modular application architecture

## Current localhost MVP

One FastAPI process serves the plain dashboard and same-origin API. It loads
the trusted saved NEV pipeline once at startup, using
[FastAPI's lifespan mechanism](https://fastapi.tiangolo.com/advanced/events/).
The registry checks feature order, class mapping, Random Forest type and
scikit-learn version compatibility. Nothing invokes training at startup or
request time. The artifact-hash prefix identifies the loaded model version.
For the approved collaboration checkpoint, the existing 2.60 MiB inference
artifact is intentionally included in Git; raw training data is not. Python
3.13 and pinned model/web dependencies make the setup repeatable without
Mac-specific paths. See `models/experiment_2/INFERENCE_ARTIFACT.md`.

`POST /predict-fault` validates one strict JSON observation, restores the
canonical seven-column order, calls `src/prediction/nev_fault.py`, and returns
the label and class probabilities. `POST /analyze-csv` validates every uploaded
observation first, then uses the same inference function. The upload follows
[FastAPI's UploadFile interface](https://fastapi.tiangolo.com/tutorial/request-files/).
No uploaded artifact can replace the model. `/health` exposes readiness; a
missing/incompatible saved model returns 503 for inference, never retraining.

The dashboard uses only HTML/CSS/JavaScript. RUL/risk controls are disabled.
Confidence is uncalibrated class probability, not a risk score. The API is
localhost-only by default: authentication, production body-size enforcement,
calibration, safety assessment and public deployment are outside this MVP.
The 5 MiB application limit is checked after multipart parsing; it is not a
replacement for a production ingress limit. Uploads are not retained, although
the framework may use a temporary spool file during request processing.

## Future specialized models

| Module | Own inputs and preprocessing | Output / status |
| --- | --- | --- |
| NEV fault diagnosis | Seven already-normalized sensors; saved NEV pipeline | Normal / Motor / Inverter / Battery class and probabilities; active |
| Motor RUL | Approved motor degradation history, identity and operating conditions | Remaining life with documented unit; contract only |
| Battery RUL | Approved battery degradation history and its own failure definition | Remaining life with documented unit; not connected |
| Bearing/drivetrain RUL | Approved component history, loads and end-of-life definition | Remaining life with documented unit; not connected |

Future routing should use an explicit module/component identifier, not guess
which model applies from whatever columns happen to be present. Each adapter
owns its feature allowlist/order, schema validation, training-only fitted
preprocessing, trusted artifact/version, supported domain, output units and
coverage evidence. The common API can dispatch to an approved adapter; the
dashboard must label each output's component, data source, observation time or
cycle, units and availability separately. Unavailable modules stay unavailable.
No dynamic plugin system or additional component endpoint is implemented now.

`src/rul/contracts.py` defines `RULModel` and `RULPrediction`. The required
estimated RUL has **no numerical default**. Observation time is optional for
cycle-only histories; a date must not be invented. A future adapter must verify
that history belongs to one machine and uses observations at or before the
requested prediction point. Contracts alone do not enforce the future
dataset's sequencing, identity or validated performance.

## Data compatibility and multi-EV limits

- Never join unrelated datasets by row position or pretend they describe the
  same vehicle. A valid relationship requires documented identifiers,
  compatible measurement times, provenance and meaning.
- Preserve each model's native units and normalization. NEV physical-unit
  normalization parameters are unknown, so raw readings cannot safely be
  converted into NEV inputs by guessed scaling.
- NEV lacks timestamps and vehicle IDs. Current CSV rows are independent
  observations in uploaded order; they are not a longitudinal trajectory.
- Motor/battery/bearing RUL needs separately approved degradation data and
  leakage-safe grouped/time-aware evaluation. Fault diagnosis does not imply
  remaining-life prediction.
- Supporting BEVs, PHEVs, FCEVs or additional manufacturers requires relevant
  training data and independent validation for those domains, compatible
  sensing/units and documented component failure definitions. The current
  model is not universal and does not establish real-world OEM performance.

Experiment 1 and Experiment 2 remain separate, reproducible evidence. Their
reports and metrics are not overwritten by the application. Future dataset
approval and model training are independent review steps.
