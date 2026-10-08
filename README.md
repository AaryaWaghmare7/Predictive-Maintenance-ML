# PREDICTIVE MAINTENANCE - ML

A beginner-friendly machine-learning project that uses EV sensor data to
support predictive-maintenance analysis. The project focuses on a repeatable
workflow: inspect data, clean it, explore it, choose a target, and only then
train and evaluate a model.

> Tesla may be used as an industry case study only. This project uses no Tesla
> proprietary data.

## Project status

| Stage | Status |
| --- | --- |
| Experiment 1: inspection, preprocessing and EDA | Complete; retained as a negative/weak-signal experiment |
| Experiment 1, Model 1: Failure_Probability | Logistic Regression baseline, Random Forest notebook experiment, and signal investigation complete; paused |
| Experiment 1, Model 2: Maintenance_Type | Signal investigation complete; no classifier trained |
| Experiment 2: NEV Fault Label | Proof-of-concept fault-diagnosis module complete; reviewed and approved for the repository |
| Localhost application | Saved-model API, manual dashboard and CSV analysis approved for a collaboration checkpoint |
| Remaining Useful Life / risk | RUL contracts and dataset requirements only; no model or numerical predictions |

The Experiment 1 dataset contains 175,393 records and 30 columns. The first
preprocessing run found no missing values and no exact duplicate rows.
Experiment 2 uses 11,000 observations and seven normalized sensor features
for four-class fault **diagnosis**, not Remaining Useful Life (RUL) prediction.

## Setup

Use your own Python environment. The shared VS Code settings do not force an
interpreter path, Conda, or a package manager. In VS Code, run **Python: Select
Interpreter** from the Command Palette and choose your local environment.
For notebooks, also choose that environment in the kernel selector. Do not
commit a personal absolute interpreter path to workspace settings.

If creating a new environment, use these commands from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Keep original datasets in `data/raw/`. Cleaned datasets are written to
`data/processed/`. Both folders are ignored by Git, so large data files are
kept local and never pushed to GitHub.

Every collaborator must separately download or copy the same
EVIoT-PredictiveMaint dataset from Kaggle and place the original file at:

```text
data/raw/EV_Predictive_Maintenance_Dataset_15min.csv
```

GitHub stores only `data/raw/.gitkeep`, not the CSV. Pulling the repository
does not download the dataset. Keep its title row and actual header unchanged;
the project loader handles the title row. The source is documented under
CC BY-NC-SA 4.0; do not publish the raw data without checking the intended use
against the license terms.

To inspect the first dataset in `data/raw/`:

```powershell
python scripts/inspect_dataset.py
```

## Experiment 1 preprocessing

The preprocessing script performs safe, repeatable cleaning:

- standardizes column names to `snake_case`
- converts and sorts the `timestamp` column
- replaces infinite numeric values with missing values
- removes exact duplicate rows
- stops if missing values remain, so an imputation choice is made explicitly
- writes a JSON data-quality report

Run it with a dataset already placed in `data/raw/`:

```powershell
python scripts/preprocess_dataset.py
```

Or provide an explicit file path:

```powershell
python scripts/preprocess_dataset.py --input "C:\path\to\EV_Predictive_Maintenance_Dataset_CSV.csv"
```

Outputs:

- `data/processed/ev_predictive_maintenance_cleaned.csv`
- `reports/metrics/preprocessing_report.json`

## Experiment 1: Model 1 baseline

Model 1 predicts `Failure_Probability`, where 0 means No Failure and 1 means Failure. It uses the fixed 24-column operational feature set and a chronological 80/20 split. The first baseline compares a majority-class reference with balanced Logistic Regression. See `reports/05_Baseline_Model_Report.md` for the full test-set evaluation.

The recorded Logistic Regression results are ROC-AUC 0.4912 and Average
Precision 0.0966. The failure-signal investigation in
`reports/06_Failure_Signal_Analysis.md` found extremely weak signal in the
approved raw features, and Model 1 is paused.

Tejaswini's existing `notebooks/modeling/02_baseline_model.ipynb` also contains
a Random Forest experiment. Its saved validation results are ROC-AUC 0.502
and Average Precision 0.099. It uses an earlier/later validation split within
the training partition; these are validation results, not final test results.
The notebook and its saved outputs are retained as completed experimental
work. Its dataset path is resolved relative to the repository on both macOS
and Windows. It expects the local processed CSV listed in the preprocessing
section, which must be generated separately from the original raw CSV.
These results are retained without retraining during Experiment 2.

For reference, the existing baseline entry point is:

```powershell
python scripts/train_baseline.py
```

## Experiment 1: Model 2 status

Model 2 uses `Maintenance_Type`: 0 = None, 1 = Preventive, 2 = Corrective, and 3 = Predictive. Its signal investigation uses the same fixed 24 operational features as Model 1 and excludes `Timestamp`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score` to avoid leakage. See `reports/07_Maintenance_Type_Signal_Analysis.md` before authorizing a multiclass baseline.

The investigation found extremely weak signal. No Model 2 classifier has been
trained. Experiment 1 remains paused and its original conclusions are preserved.

The completed Model 1 and Model 2 signal plots in `reports/figures/` are tracked
alongside their reports. Experiment 1 generated data, metrics, and model
artifacts remain local and ignored by Git.

## Experiment 2: NEV fault diagnosis

Target: `Fault Label`. Mapping supplied by the user:
**0 = Normal, 1 = Motor Fault, 2 = Inverter Fault, 3 = Battery Fault**.
This is a separate experiment, not Experiment 1's Maintenance_Type Model 2.
Experiment 2 is a **proof-of-concept EV fault-diagnosis module**.

Each collaborator must separately supply the same three original files:

```text
data/raw/experiment_2/NEV_fault_dataset.csv
data/raw/experiment_2/NEV_fault_training_dataset.csv
data/raw/experiment_2/NEV_fault_testing_dataset.csv
```

The full file is for descriptive EDA/audit. The supplied **7,700 training /
3,300 testing** membership is preserved; there is no new random train/test
split. Five-fold stratified validation is confined to training. Raw NEV files
are ignored by Git. Their own source/license and normalization protocol still
need confirmation; Experiment 1's license must not be assumed to apply.

Inputs, in the inference order, are `Voltage (V)`, `Current (A)`,
`Motor Speed (RPM)`, `Temperature (°C)`, `Vibration (g)`,
`Ambient Temp (°C)`, `Humidity (%)`. Despite the physical units in these names,
the numbers are already normalized to approximately [0,1]. Original
physical-unit normalization parameters are not currently known, so no
physical-unit conversion is available. `Fault Label` never enters X.

| Fixed candidate | Final test accuracy | Final test macro F1 | Macro OvR ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.980909 | 0.975324 | 0.999191 |
| Random Forest | 0.994242 | 0.992082 | 0.999698 |
| HistGradientBoosting | 0.993333 | 0.990832 | 0.999759 |

**Random Forest was selected before test evaluation**, using training-only
CV macro F1. The full inference pipeline is local at
`models/experiment_2/nev_fault_pipeline.joblib`, with `metadata.json` beside
it. Metadata records feature order, class mapping, versions, fingerprints,
configurations, CV scores and final test results. For the user-approved
localhost collaboration checkpoint, **this specific fitted pipeline is now
intentionally tracked**, alongside metadata: **2,729,762 bytes (2.60 MiB)**.
This lets Tejaswini pull and run inference without Aarya's local files or raw
training datasets. Other model binaries and all raw datasets remain ignored.
See [artifact provenance, checksum and compatibility](models/experiment_2/INFERENCE_ARTIFACT.md).
Earlier experiment reports describe the original local-only storage policy;
the checkpoint changes storage, not model parameters, results or conclusions.

Reports:

- [Dataset audit](reports/08_NEV_Dataset_Audit.md)
- [EDA and leakage/signal review](reports/09_NEV_EDA_Report.md)
- [Model comparison, confusion matrices and interpretation](reports/10_NEV_Model_Comparison.md)

The approved full-run command is:

```powershell
python scripts/run_nev_experiment.py
python -m pytest -q
```

Do not repeatedly rerun/tune against the observed test results. No SMOTE,
feature engineering or hyperparameter/threshold search was performed.
The eight report figures and four statistical/metric snapshots under
`reports/figures/experiment_2/` and `reports/metrics/experiment_2/` are tracked
with the approved reports. They can be reproduced through the script, which
also retrains models; do not run it merely to receive the latest code.

Local inference with the saved pipeline (from the repository root):

```python
import joblib
from src.prediction.nev_fault import predict_nev_faults

pipeline = joblib.load("models/experiment_2/nev_fault_pipeline.joblib")
# sensors: DataFrame with exactly the seven normalized input columns above.
predictions = predict_nev_faults(pipeline, sensors)
```

Very sharp class-specific ranges and a depth-3 diagnostic tree's CV macro F1
of approximately 0.979 suggest unusually easy, rule-like class boundaries.
The high Random Forest scores describe performance on this supplied dataset,
**not evidence of equivalent real-world OEM performance**.
Realism/label-generation provenance is
unconfirmed. No timestamps or vehicle IDs are present, so the scores establish
neither temporal nor cross-vehicle generalization. Original normalization may
also have used full-dataset information; our own pipeline fitting is training
only. The model does not accept raw volts/amps/RPM by guessing a conversion.

Experiment 2 **does not provide Remaining Useful Life (RUL) prediction**.
RUL will be implemented as a separate module using an appropriate
degradation/time-to-failure dataset. The localhost application below uses the
existing saved fault model; it does not add a new experiment or retrain it.
Experiment 2 has been reviewed and approved for repository publication;
this approval does not authorize new modeling experiments.

## Running the Localhost MVP

The **EV Predictive Maintenance Platform** is a minimal FastAPI backend with
plain HTML/CSS/JavaScript, served from the same origin. It is a research MVP,
not a public deployment or a vehicle safety system. No frontend build step,
database or external UI framework is needed.

**Currently implemented:** fault diagnosis, manual normalized-input form,
CSV batch analysis and model confidence display.

**Not yet implemented:** RUL prediction, final risk engine, production/raw OEM
telemetry normalization or cloud deployment. No fake RUL or risk values exist.

Use **Python 3.13**, matching the saved artifact's Python 3.13.9 environment.
The inference libraries and web dependencies are pinned in `requirements.txt`.
Raw CSV datasets are **not required** to run this application; the trusted
pipeline is included in this checkpoint. Never rerun training just to start it.

First open a terminal in your existing repository clone. If `git status`
shows local changes, preserve legitimate work on a personal branch before
switching/pulling. Do not discard work or bypass a divergent-history error.

### macOS / Linux

Use your already-selected Python 3.13 Conda environment, or create a virtual
environment below (`python3` must refer to Python 3.13). If the correct
environment already exists, activate it instead of recreating it.

```bash
git status
git switch main
git pull --ff-only origin main
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_nev_inference.py
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8010 --reload
```

### Windows / Tejaswini testing notes

Run these from the repository root in PowerShell. Use your own interpreter;
the shared VS Code settings do not force a Mac path or Conda. If Python 3.13
is not installed, install/select that version first. If a suitable `.venv`
already exists, skip creation and activate it.

```powershell
git status
git switch main
git pull --ff-only origin main
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/smoke_test_nev_inference.py
python -m pytest -q
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8010 --reload
```

If PowerShell blocks activation, you can use
`.\.venv\Scripts\python.exe` in place of `python` in these commands without
changing execution policy. Select this same environment in VS Code locally.

If pytest encounters **WinError 5** for its temporary directory, use a fresh
temporary path (this is not a model-code failure):

```powershell
$testTemp = Join-Path $env:TEMP ("ev-pytest-" + [guid]::NewGuid().ToString("N"))
python -m pytest -q --basetemp="$testTemp"
```

Open [the dashboard](http://127.0.0.1:8010) or
[interactive API documentation](http://127.0.0.1:8010/docs).
Port 8010 avoids the applications already using 8000/8001 on Aarya's Mac.
If 8010 is also occupied, choose another unused port and update the URL.
Stop your application with Ctrl+C. Keep it bound to localhost; authentication,
production upload controls, monitoring and deployment hardening are not
implemented.

The app loads the trusted local
`models/experiment_2/nev_fault_pipeline.joblib` once at startup. Use the pinned
dependencies recorded in `models/experiment_2/metadata.json`. Never load an
untrusted joblib/pickle file. A missing, invalid or version-incompatible
pipeline leaves the dashboard available and health status degraded;
prediction returns HTTP 503. Verify the tracked artifact was pulled and the
pinned dependencies were installed, then restart. **No automatic retraining
is performed.**
The smoke test checks synthetic inputs, column reordering, class mapping and
probabilities without fitting anything. It is not a new model evaluation.

### API and expected inputs

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Application/model readiness, artifact fingerprint, disabled RUL/risk status |
| `POST /predict-fault` | One observation with exactly seven normalized sensor fields |
| `POST /analyze-csv` | Multipart upload named `file`; validate all rows, then batch diagnosis |
| `GET /` | Localhost dashboard |

**Prototype mode: inputs are normalized values approximately in [0,1].**
Original physical-unit normalization parameters are unknown. Despite the
original column names, do not send raw volts, amps, RPM or degrees Celsius.
JSON must use the exact feature names; order does not matter because the
backend restores the saved model's seven-column order. Missing/extra fields,
numeric strings, booleans, nulls, NaN/Infinity and out-of-range values are
rejected. The existing 1e-12 boundary tolerance is retained, without clipping.

Example request body (user-supplied Motor Fault smoke-test values, not raw OEM
measurements or a new labeled evaluation dataset):

```json
{
  "Voltage (V)": 0.805,
  "Current (A)": 0.597,
  "Motor Speed (RPM)": 0.073,
  "Temperature (°C)": 0.180,
  "Vibration (g)": 0.921,
  "Ambient Temp (°C)": 0.128,
  "Humidity (%)": 0.125
}
```

The response includes `predicted_class`, `predicted_fault`,
`model_confidence`, all four `class_probabilities`, prediction `status`,
`model_version` (artifact-hash prefix) and
`input_mode: "normalized_proof_of_concept"`.
**Confidence is the uncalibrated probability of the predicted class, not a
real-world failure probability, future-failure forecast or risk score.**
Even a confident Normal label is not a guarantee of health or safe operation.

### Four manual test examples

Enter all seven numbers from one row, then click **Analyze Vehicle**. These
are the user's normalized proof-of-concept test examples, **not raw physical
measurements**. The expected class codes are 0 / 1 / 2 / 3. Confidence remains
an uncalibrated class probability, even when it displays 100%.

| Expected class | Voltage | Current | Motor Speed | Temperature | Vibration | Ambient Temp | Humidity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 — Normal | 0.939 | 0.200 | 0.544 | 0.240 | 0.301 | 0.582 | 0.397 |
| 1 — Motor Fault | 0.805 | 0.597 | 0.073 | 0.180 | 0.921 | 0.128 | 0.125 |
| 2 — Inverter Fault | 0.586 | 0.275 | 0.355 | 0.398 | 0.452 | 0.728 | 0.326 |
| 3 — Battery Fault | 0.180 | 0.453 | 0.393 | 0.782 | 0.227 | 0.287 | 0.515 |

With the server running, a second terminal in the same Python environment can
verify these four examples, invalid-input rejection, dashboard assets and CSV
upload over HTTP:

```bash
python scripts/smoke_test_localhost.py
```

### Dashboard and CSV upload

The dashboard offers the seven-field manual form, **Analyze Vehicle**, a
result card, CSV upload, class counts/percentages, dominant abnormal labels
(including ties), mean model confidence and row-level predictions.
Synthetic examples are clearly marked and can be downloaded to try the UI.
Changing inputs clears stale results.

CSV must be UTF-8 (a UTF-8 BOM is accepted), with exactly these seven unique
headers in any order:

```text
Voltage (V),Current (A),Motor Speed (RPM),Temperature (°C),Vibration (g),Ambient Temp (°C),Humidity (%)
```

No label, timestamp or vehicle-ID column is accepted in the current upload
schema. All rows need finite normalized values. Invalid uploads are rejected
in full, not silently cleaned. Limits: **5 MiB**, **20,000 observations**;
the response returns at most the first **1,000 row predictions**, while class
counts and mean confidence cover every validated row. Limit failures use
HTTP 413, invalid data uses 422, and model unavailability uses 503.
The application does not retain uploads; the upload framework may spool a
temporary file while receiving the request. Rows are independent observations
in file order, **not a temporal sequence**. No longitudinal health trajectory
or cross-vehicle generalization is inferred.

### Future RUL and specialized component models

The dashboard shows **“RUL model not yet connected.”** and disabled risk
controls. There is no RUL/risk endpoint, fitted model, fake remaining-life
estimate or fabricated score. `src/rul/` contains contracts only, with
separate `data/rul/`, `models/rul/` and `reports/rul/` locations.
A future prediction must include component, machine ID, actual estimated RUL,
documented unit and model version; no numerical default is provided.

See [RUL dataset requirements](reports/11_RUL_Dataset_Requirements.md) before
manually approving any new dataset, and
[modular architecture](docs/MODULAR_ARCHITECTURE.md) for separate fault,
motor, battery and bearing/drivetrain adapters. NEV does not support RUL.
BEV/PHEV/FCEV or manufacturer coverage requires appropriate training and
independent validation data; the current model is not universal.

Run all tests without regenerating the production model:

```bash
python -m pytest -q
```

Saved-artifact integration tests now require the tracked pipeline and check
the recorded model dependency versions, four supplied examples and loading
independently of the working directory. Missing artifacts fail, not silently
skip. API unit tests use tiny synthetic fixtures; no test automatically
regenerates the NEV production artifact. Existing Experiment 1 and Experiment 2
reports, final metrics and raw data are preserved. Windows runtime validation
must still be confirmed on Tejaswini's actual laptop; Mac tests alone do not
prove every Windows environment works.

## Collaboration workflow

To receive the latest code in an existing clone, first check for local changes:

```powershell
git status
git switch main
git pull --ff-only origin main
python -m pytest -q
```

Commit legitimate local work on your personal branch before switching branches
or pulling. If `--ff-only` reports divergent history, preserve your commits and
resolve the divergence together instead of force-pushing or resetting files.
The dataset and local Python environment must be supplied separately as above.

Do all work on a personal branch, then open a pull request for review:

```powershell
git checkout -b tejaswini/feature-name
git add src scripts tests README.md
git commit -m "Add preprocessing pipeline"
git push -u origin tejaswini/feature-name
```

## Project layout

```text
data/raw/          Original local datasets
data/processed/    Local cleaned datasets
data/rul/          Future approved degradation dataset; no dataset selected
notebooks/eda/     Exploratory data analysis notebooks
notebooks/modeling/Model experiments
scripts/           Runnable commands
src/preprocessing/ Reusable cleaning and validation code
src/api/           FastAPI backend and static localhost dashboard
src/rul/           Future RUL interfaces only
models/rul/        Future RUL artifacts; none trained
reports/rul/       Future RUL evidence and evaluation
reports/metrics/   Local preprocessing and model reports
tests/             Automated checks for reusable code
```
