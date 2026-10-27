# EVision

AI-Powered EV Predictive Maintenance · Predict. Prevent. Drive Further.

EVision is the application brand; the repository remains
`AaryaWaghmare7/Predictive-Maintenance-ML`. Existing research modules, experiments
and dataset conclusions retain their original scope.

An EV sensor-data research project by **Aarya Waghmare and Tejaswini**, combining
dataset assessment, reproducible classification experiments and an interactive
fault-diagnosis dashboard.

**Implemented:** four-class fault diagnosis, manual normalized inputs, validated
CSV batch inference and interactive prediction summaries.
**Under Development:** component/bearing Remaining Useful Life (RUL).
**Disabled:** unsupported maintenance-risk assessment.

This is a research proof of concept, **not a published paper**, OEM-validated
diagnostic product or vehicle safety certification. It uses no Tesla proprietary
data. No cloud deployment is currently claimed.

![Fault-diagnosis dashboard](reports/figures/showcase/dashboard.png)

## Research findings

### Experiment 1 — a useful negative result

The EVIoT-PredictiveMaint dataset contains 175,393 observations and 30 columns.
With the fixed 24-feature allowlist and a chronological 80/20 split, balanced
Logistic Regression achieved ROC-AUC **0.4912** and Average Precision **0.0966**.
The retained Random Forest notebook experiment also showed near-chance
discrimination: **validation** ROC-AUC 0.502 / Average Precision 0.099.

A separate Maintenance_Type signal investigation found extremely weak
relationships; no classifier was trained for that target. Both investigations
are preserved rather than hidden or replaced by the more successful dataset.

The balanced Logistic Regression's recorded test metrics are retained here
alongside the remote README's research summary:

| Metric | Supplied test result |
| --- | ---: |
| Accuracy | 50.05% |
| Failure precision | 9.76% |
| Failure recall | 48.91% |
| Failure F1 | 16.27% |
| ROC-AUC | 0.4912 |

See [baseline](reports/05_Baseline_Model_Report.md),
[failure-signal investigation](reports/06_Failure_Signal_Analysis.md) and
[maintenance-type investigation](reports/07_Maintenance_Type_Signal_Analysis.md).

### Experiment 2 — NEV fault diagnosis

The supplied split is preserved: **7,700 training / 3,300 testing observations**.
Five-fold cross-validation confined to training selected Random Forest by mean
macro F1 **before** final test evaluation. Learned preprocessing remains inside
each pipeline. No new train/test split, SMOTE or test-driven tuning was used.

| Model | Test accuracy | Test macro F1 | Test macro OvR ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.980909 | 0.975324 | 0.999191 |
| **Random Forest** | **0.994242** | **0.992082** | **0.999698** |
| HistGradientBoosting | 0.993333 | 0.990832 | 0.999759 |

The selected classifier uses 200 trees and balanced class weights. The complete
named-column preprocessing/inference pipeline is saved and intentionally tracked
at `models/experiment_2/nev_fault_pipeline.joblib` (about 2.60 MiB). Loading and
evaluating this unchanged artifact reproduces the recorded Random Forest scores
on the 3,300 supplied test rows; **no retraining is needed to run the app**.

Its confusion matrix uses actual rows / predicted columns in class order 0–3:

```text
[[1500,   0,   0,   0],
 [   0, 600,   0,   0],
 [   0,   0, 599,   1],
 [   0,   0,  18, 582]]
```

These 19 errors matter: most are Battery Fault observations classified as
Inverter Fault. High aggregate accuracy does not eliminate class-specific errors.

See the [dataset audit](reports/08_NEV_Dataset_Audit.md),
[EDA](reports/09_NEV_EDA_Report.md),
[model comparison](reports/10_NEV_Model_Comparison.md) and
[recorded metadata](models/experiment_2/metadata.json). Only the selected saved
classifier was re-evaluated for the showcase audit; other candidate scores are
preserved historical results.

## Capabilities and architecture

One FastAPI process serves the HTML/CSS/JavaScript dashboard and its API.
Both manual and CSV inputs pass through shared validation and the same trusted
saved pipeline. No training runs during startup or requests. Paths are resolved
relative to the project, not an individual computer.

| Layer | Existing implementation |
| --- | --- |
| Dashboard | `src/api/static/`: EVision branding, red/black studio, local transparent WebP car, responsive/keyboard-accessible controls |
| Validation/API | `src/api/`: strict inputs, multipart parsing, bounded requests and useful errors |
| Inference | `src/prediction/nev_fault.py`, `src/storage/model_registry.py`: trusted artifact, feature/class/version checks |
| Research evidence | `src/eda/`, `src/training/`, `src/evaluation/`, notebooks and numbered reports |

The dashboard includes four fill-only demo samples, explicit Analyze actions,
busy/recovery states, a confidence display and a downloadable tiny sample CSV.
The automotive landing page links to Diagnostics, Analytics, Models, Reports
and Documentation. Explore Demo fills a genuine supplied sample observation;
it never invents operational records or runs inference without Analyze.
**Model Confidence is an uncalibrated predicted-class probability**, not a
future failure probability or risk score.

The CSV donut uses whole-file prediction counts, with no fabricated empty-state
statistics. Select a class legend with mouse
or keyboard to filter the row table; select it again or Show all classes to
clear. Whole-file counts never change when filtering. Row details are capped and
labeled accordingly; original observation numbers are preserved. Download
predictions CSV exports only the actual returned predictions (up to 1,000),
not all rows when details were capped, regardless of the active display filter.
It includes predicted classes and confidence, not ground-truth diagnoses.

![CSV prediction analytics](reports/figures/showcase/telemetry.png)

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Dashboard |
| `GET /health` | Application/model status; stays available if inference is degraded |
| `GET /ready` | Model-aware readiness: HTTP 200 when ready, 503 otherwise |
| `POST /predict-fault` | One normalized observation as strict JSON |
| `POST /analyze-csv` | Multipart CSV upload using field `file` |
| `GET /docs` | Interactive API documentation |

See [architecture and future component boundaries](docs/MODULAR_ARCHITECTURE.md).

## Run locally

Use **Python 3.13**, matching the artifact's recorded Python 3.13.9 environment.
From the repository root, in your selected environment:

```bash
python -m pip install -r requirements-api.txt
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012
```

Open [the local dashboard](http://127.0.0.1:8012/). Keep the server running.
**Do not open `src/api/static/index.html` with a `file://` URL or Live Server.**
Its `/static/...` assets and same-origin API require FastAPI. Direct file previews
redirect to the documented default localhost port 8012; the server must already
be running. If you choose a different port, open that server URL explicitly.
No raw dataset, Node installation, frontend build or external font is required
for inference. The shared setup contains no personal interpreter path.
The original AI-generated car cutout is served as a local transparent WebP
(about 184 KB); it is not an OEM asset or a reference screenshot pasted into the
page. CSS coordinates the car entrance, lighting and headline; native browser
APIs drive bounded, reversible scroll movement and staggered panel reveals.
No GSAP/ScrollTrigger, Three.js, CDN or frontend build is required. Mobile views
use a shorter entrance without scroll parallax. System reduced-motion preferences
and the footer's Reduce motion control disable movement. Offscreen scroll work
is detached and observers/listeners are cleaned up on page exit; forms never
depend on motion. This is a 2D cutout, not simulated 3D rotation.

For research notebooks and all tests, install `requirements.txt` instead; it
includes the same runtime pins plus research/test dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/smoke_test_nev_inference.py
python scripts/smoke_test_localhost.py --base-url http://127.0.0.1:8012
```

The HTTP smoke script needs the running server and uploads small temporary
fixtures, not training files. Optional frontend tests use Node's built-in runner:

```bash
node --check src/api/static/app.js
node --test tests/frontend/test_dashboard.cjs
```

Unit tests use tiny synthetic fixtures; they never regenerate the production
model. Full saved-test metric reproduction is additionally checked when the
original local testing CSV is present and matches its recorded hash. A clean
checkout without raw datasets skips two optional checks: Experiment 1 raw-data
feature selection and NEV test-metric reproduction. Saved-artifact smoke tests
still run without either dataset.

### Windows PowerShell — collaboration checkpoint

Use standard **64-bit Python 3.13**, not a free-threaded build. Replace the first
path below with Tejaswini's existing clone location. Do not paste a Mac interpreter
path into VS Code or the repository. Stop if `git status` shows work you have not
saved; do not reset, discard files or force-pull to make the commands continue.

```powershell
Set-Location "C:\path\to\Predictive-Maintenance-ML"
git status --short
if (git status --porcelain) { throw "Save/review your local work before pulling." }
git fetch origin
git switch main
git pull --ff-only origin main

py -3.13 --version
if (!(Test-Path ".venv\Scripts\python.exe")) { py -3.13 -m venv .venv }
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\smoke_test_nev_inference.py
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012
```

Run each step only after the previous command succeeds. An existing `.venv` must
also use Python 3.13; if it does not, stop and select/create a compatible environment
without overwriting local work. Calling its Python directly avoids changing
PowerShell's execution policy. In VS Code, select that interpreter locally.
For inference only, `requirements-api.txt` is sufficient; `requirements.txt`
includes it plus the research/test dependencies. No new package was installed
for the animation; no Node/GSAP/CDN dependency is needed to run EVision.

Keep the server terminal open. In a second PowerShell window:

```powershell
Set-Location "C:\path\to\Predictive-Maintenance-ML"
Start-Process "http://127.0.0.1:8012/"
.\.venv\Scripts\python.exe scripts\smoke_test_localhost.py --base-url http://127.0.0.1:8012
```

Check the red/black styling, car entrance and scrolling in both directions,
all six navigation links, Get Started, Explore Demo and the four sample predictions.
Download Sample CSV, upload that same four-row file, verify one predicted row
per class, try class filtering and download the returned predictions. Check
Reduce motion and a narrow browser layout. Try a CSV missing one sensor to verify
the named validation error and recovery. Do not enter raw physical readings.

The trusted `.joblib` pipeline and local frontend assets are included in normal
Git. No raw dataset is needed for the website. Obtain original datasets separately
only for approved research reproduction; do not add CSVs/RUL archives to Git.
Native Windows execution still needs Tejaswini's confirmation; local tests and
availability of pinned Python 3.13 Windows wheels are not a Windows runtime test.

To suggest changes without overwriting `main`, create a branch after a clean pull:

```powershell
git switch -c tejaswini/evision-feedback
# Edit only the intended files, then inspect before committing.
git diff
git add path\to\the-edited-file
git diff --cached
git commit -m "fix: describe the EVision improvement"
git push -u origin tejaswini/evision-feedback
```

Open a pull request into `main` for review; do not force-push or directly replace
the shared branch. For bug reports include Windows/Python/browser versions,
reproduction steps, expected/actual behavior, screenshots and sanitized browser
console/server errors. Use the tiny sample CSV instead of private telemetry.

## Input contract

Despite physical-unit labels, values are **already normalized approximately to
[0,1]**. The original normalization parameters are unknown: do not enter raw
volts, amps, RPM or temperatures and do not invent a conversion.

```text
Voltage (V),Current (A),Motor Speed (RPM),Temperature (°C),Vibration (g),Ambient Temp (°C),Humidity (%)
```

Classes: **0 Normal · 1 Motor Fault · 2 Inverter Fault · 3 Battery Fault**.

JSON requires exactly these seven named numeric fields. CSV accepts them in any
order, supports UTF-8/BOM and surrounding header whitespace, and reports/ignores
extra columns. **Fault Label is not required and never enters inference.**
Blank/duplicate headers, missing sensors, malformed records and blank,
nonnumeric, nonfinite or invalid-range sensor values reject the entire upload.
No row is silently dropped, imputed or clipped at the API boundary.

Limits: 5 MiB CSV / 20,000 observations / first 1,000 row predictions, with
whole-file summaries. Total multipart request bytes are bounded at 6 MiB before
parsing, including chunked requests; manual request bodies are limited to 16 KiB.
Uploads are not retained by the app; the framework may spool temporary files.

## Limitations and research integrity

- The NEV class boundaries are unusually easy: a depth-3 decision tree achieved
  approximately 0.979 CV macro F1. Results describe this supplied dataset,
  **not equivalent real-world OEM performance**.
- There are no timestamps or vehicle IDs: temporal and cross-vehicle
  generalization cannot be claimed. File order is not time order.
- Original normalization, acquisition and label-generation protocols are
  unverified. Unknown upstream leakage cannot be ruled out.
- Confidence is not calibrated to real-world fleet failure risk. Normal does not
  guarantee safe operation. Feature importance is not causality.
- RUL needs verified component identities, ordered run-to-failure records,
  failure endpoints and leakage-safe component-aware evaluation. No validated
  local RUL model exists; no RUL metrics or estimates are presented.
- Risk remains disabled until compatible component outputs and defensible
  maintenance thresholds are validated. Unrelated fault and bearing datasets
  must not be joined by row position.

## Data, licensing and credit

Raw data is not tracked. Experiment 1 is the EVIoT-PredictiveMaint Dataset on
Kaggle, documented as CC BY-NC-SA 4.0; its terms do not automatically apply to
NEV. Reproduction paths are `data/raw/EV_Predictive_Maintenance_Dataset_15min.csv`
and the three supplied NEV CSVs in `data/raw/experiment_2/`.

**The NEV source/license and trained-artifact redistribution permissions remain
unverified. Resolve these before promoting/distributing this as a licensed
public release.** The already tracked artifact is preserved; no additional
dataset or model publication is authorized by this audit.

Aarya Waghmare and Tejaswini collaborate on this research project. Existing
EDA, preprocessing, modeling-notebook contributions and attribution remain in
Git history and the preserved reports. This project is not a published paper.

## Roadmap and progress

The long-term vision remains a modular platform combining fault diagnosis,
component Remaining Useful Life and evidence-based maintenance prioritization.
Only the validated capabilities below are active:

| Module | Current status |
| --- | --- |
| Experiment 1 / failure and maintenance-type signal studies | Complete; weak-signal conclusions preserved |
| Experiment 2 / Random Forest fault diagnosis | Complete; recorded supplied-test evaluation |
| EVision / FastAPI / manual and CSV diagnosis | Working localhost collaboration checkpoint; native Windows confirmation pending |
| Component/bearing RUL | Under Development; verified run-to-failure dataset/model unavailable locally |
| Maintenance Risk Engine | Disabled pending compatible outputs and defensible thresholds |
| Cloud deployment | Not deployed or approved |

Future RUL work will audit unit identities, run-to-failure endpoints and labels,
extract appropriate vibration/degradation features, compare regression pipelines
with component-aware splits, and evaluate MAE, RMSE, R² and near-failure errors
before integration. It requires separate approval and suitable data. Estimates
will concern the supported component, not the entire vehicle.

## Deployment recommendation — not deployed

Prefer **one Render web service for FastAPI and the static frontend**. Relative
API URLs remain same-origin; no frontend API-base environment setting or wildcard
CORS is needed. A Render + Vercel split would introduce another origin, explicit
CORS configuration and independent releases with no current requirement for them.

Proposed settings, to apply **only after approval**:

- Python: `PYTHON_VERSION=3.13.9`.
- Build: `python -m pip install -r requirements-api.txt`.
- Start: `python -m uvicorn src.api.app:app --host 0.0.0.0 --port "$PORT" --workers 1`.
- Health check: `/ready`; include the trusted artifact, not raw datasets.
- Do not enable automatic deployments before approval.

These settings follow [Render's web-service requirements](https://render.com/docs/web-services),
[Python version configuration](https://render.com/docs/python-version) and
[HTTP health checks](https://render.com/docs/health-checks).
A Linux build/resource test, dependency vulnerability review, ingress/rate limits
and a public-service threat review still precede deployment. Basic response
headers and per-request caps do not make this an automotive safety system.

See the [showcase audit and unposted LinkedIn research summary](reports/13_Showcase_Audit.md)
for current test evidence, release blockers, Git state and exact changed files.
