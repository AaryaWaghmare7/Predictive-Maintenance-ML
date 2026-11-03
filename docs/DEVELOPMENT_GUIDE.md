# EVision — Development and Collaboration Guide

Detailed setup, input contracts, security boundaries and contributor backgrounds.
The root README provides the concise project overview. Historical research reports
remain unchanged.

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
node --test tests/frontend/test_dashboard.cjs tests/frontend/test_analytics.cjs
```

Unit tests use tiny synthetic fixtures; they never regenerate the production
model. Full saved-test metric reproduction is additionally checked when the
original local testing CSV is present and matches its recorded hash. Absence of
that private CSV skips only that reproduction test, not saved-artifact smoke tests.

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

## Reliability, dataset workflows and security


The saved classifier is unchanged. Explore Demo intentionally fills the Motor
Fault sample. A displayed 100% is that model's uncalibrated predicted-class
confidence, not an observed failure rate or real-world fault probability. The
four supplied samples predict all four different classes; there is no Motor
fallback on invalid data. The user's exact case `[0.80, 0.597, 0.073, 0.10, 0.92,
0.28, 0.12]` was reproduced through the pipeline, API and browser as Motor with
`[0,1,0,0]` class probabilities. All 200 trees agree; no preprocessing changes
these complete inputs. This reflects sharp supplied-dataset boundaries, not
diagnostic certainty. Exact-case regression tests preserve the behavior rather
than changing displayed confidence cosmetically. Details are in report 14.

**Dataset Upload & Evaluation** adds three explicit workflows using
`POST /analyze-dataset` (multipart fields `file`, `mode`, `table`, `mapping`,
`normalized`, `labels_confirmed`):

- Inspect: lists recognized/missing/extra columns, target availability, numeric
  ranges, dtypes, duplicates and invalid cells. No inference runs.
- Predict: requires explicit confirmation of the same NEV normalization/context.
  Names or 0–1 ranges alone are insufficient. No accuracy is measured.
- Evaluate: additionally requires a confirmed `Fault Label` with the documented
  0/1/2/3 meaning. Shows per-class scores, macro precision/recall/F1, confusion
  matrix, applicable AUC/AP and probability diagnostics. These are uploaded-data
  scores, **not independent unseen-test performance**. Exact known original
  training/full file hashes are blocked from evaluation; reformatted files,
  subsets, duplicate lineage and dishonest attestations cannot generally be
  detected. No model is trained, tuned or calibrated.

Supported files: UTF-8 CSV/BOM with comma, semicolon, tab or pipe separators,
and SQLite database files containing explicitly selected ordinary tables.
SQLite is imported into memory, query-only, with restricted schema/features,
SQL authorizer, limits and a two-second query budget. Uploaded SQL is never
executed; no remote database connection is supported. A Python build lacking
SQLite `deserialize` must use CSV exports. Keep Python/SQLite patched; this is
not an operating-system sandbox for arbitrary hostile databases.

The UI handles at most five files **sequentially and separately**, each limited
to 5 MiB, 20,000 rows, 100 columns and 128-character headers. There is no merge,
row deletion, implicit unit conversion or automatic retraining. An optional
mapping is an explicit JSON object from canonical model feature to source
column; it cannot map labels or override an existing canonical feature.
Errors identify missing sensors or invalid observations/columns. Existing
`/analyze-csv` remains the strict comma-CSV prediction-only compatibility route.

Manual results now show observed normalized readings and a versioned educational
inspection guide from `src/prediction/fault_guidance.json`. It is not causal
feature attribution or a confirmed diagnosis. General EV safety guidance follows
[NHTSA](https://www.nhtsa.gov/vehicle-safety/electric-and-hybrid-vehicles);
no DIY high-voltage repair procedures are provided. A non-Normal label requests
technician review, while Normal retains routine review. These are **inspection
workflow categories**, not validated urgency or Low/Medium/High failure risk.
OEM limits, a time horizon and compatible validated component data are missing.
RUL and quantified risk remain unavailable.

### Security configuration — no deployment performed

Local loopback use needs no secrets or accounts. Existing macOS and Windows
commands above remain valid after `python -m pip install -r requirements.txt`.
AnyIO 4.14.2 and idna 3.15 are targeted security pins; ML package versions remain
unchanged. No new framework or database-service package is required.

`.env.example` contains placeholders only; **it is not automatically loaded**.
For optional access protection set `EVISION_API_TOKEN` in the process environment
to a securely generated random ASCII token of at least 32 characters. The UI
then shows a password-style token field. It is kept only in page memory, sent
as a Bearer header and cleared by clearing the field/reloading; never include
it in screenshots, URLs, Git or bug reports. No default token is supplied.

Production mode (`EVISION_MODE=production`) additionally requires explicit
`EVISION_ALLOWED_HOSTS` (comma-separated hostnames, no scheme/port) and refuses
startup without the token. Every POST is then protected and interactive docs/
OpenAPI are disabled. No accounts, passwords, cookies, roles, private history or
admin operations are implemented. This is a **shared single-operator token gate**,
not a multi-user authorization/session solution; rotation/revocation is manual
through the environment and server restart. Do not use it for private multi-user
records. Add a separately reviewed identity provider before that scope exists.

Same-origin requests only; arbitrary Origin/Host and wrong content types are
rejected. CSP restricts application scripts to local files and the exact preview
guard hash. Inline styles remain allowed because existing animation/chart code
sets dynamic CSS properties; inline scripts are not broadly allowed. Local-only
Swagger needs its narrowly scoped jsDelivr dependency and an exact script hash.
No user upload/model/dataset is served by the static mount. The canonical joblib
hash is checked **before** deserialization; only trusted backend-controlled
artifacts are permitted. A hash is not a substitute for a trusted supply chain.

Limits are 60 manual requests/minute/client and 20 upload requests/minute/client,
with two concurrent uploads per process, 16 KiB JSON and 6 MiB total multipart
bodies. HTTP 429 includes Retry-After. These **in-memory single-worker controls**
are not distributed rate limiting or a hard CPU/memory sandbox. Production still
requires approved TLS, trusted proxy configuration, ingress limits, shared rate
limiting when applicable and independent security review. HSTS is emitted only
for production requests correctly identified as HTTPS, never localhost HTTP.

See [the detailed reliability/security audit](../reports/14_Reliability_and_Security_Audit.md)
for original/final metrics, dependency findings, test evidence and remaining
limitations. Historical results/reports are preserved. Training commands in
earlier research reports are **not** part of application startup; do not rerun
`scripts/run_nev_experiment.py` just to run EVision. These changes are part of the GitHub collaboration checkpoint. No deployment,
new model fitting or RUL-data download is required.

## Interactive analytics

Open **Analytics** in the navigation. Choose either the clearly labeled original
NEV reference or an actual uploaded file; original charts are never presented as
your vehicle's telemetry. The seven sensor histograms, class boxplots, Pearson
heatmap, class-colored feature scatter, association and saved-model importance
charts are interactive SVGs with source/class filters and numeric tooltips.
Scatter range controls provide zoom; Reset restores data bounds. No charting
framework, CDN, extra dependency or model fitting is required.

Both CSV prediction and Dataset Upload & Evaluation feed this same dashboard.
Inspect-only data can be explored without predicting: raw-scale values retain
their actual histogram ranges and remain ineligible for NEV inference without
compatible normalization. Missing/nonfinite chart exclusions are counted. Class
charts use explicitly labeled **model predictions** or **user-confirmed labels**;
unlabeled observations are not assigned invented categories. Multiple files keep
separate source entries, even when filenames are identical; replacing a selection
clears its previous charts. No files are combined.

Histograms, quartiles and correlations use **all available upload observations**,
not the first 1,000 returned predictions. Scatter is limited to a deterministic
class-stratified display sample of at most 1,000 points and is labeled accordingly.
Original training scatter uses joint-bin counts rather than raw row records.
Full-data reference histograms/correlation use 11,000 rows; original class-based
plots/associations use the 7,700 training rows only. Importance values come from
the saved historical analysis and are **not upload-specific or causal**.

Upload-versus-training shows observed ranges, medians, out-of-training-range
counts and histogram differences. Range flags are not physical safety limits.
Shape review uses total variation >0.25 with at least 30 finite observations,
an explicitly documented **review heuristic, not a calibrated OOD detector or
statistical significance test**. Class mix and incompatible scales may explain
differences; no normalization parameters are invented and no rows are clipped.

Download analytics JSON, source-labeled summary CSV, or the current chart view as
a standalone HTML report. Open the HTML in a browser and print to PDF if needed;
no server-side PDF service or private report storage is introduced.

`GET /analytics/reference` serves a small aggregate snapshot in
`reports/metrics/experiment_2/dashboard_reference.json`, included in the repository. It contains no raw training vectors. To reproduce it with the
original local raw files, run `python scripts/build_analytics_reference.py`, then
restart FastAPI. This verifies recorded file hashes and never trains a model.
If the snapshot is missing, reference/importance/comparison states are unavailable
instead of being fabricated; upload analytics can still work.

See [interactive analytics verification](../reports/15_Interactive_Analytics.md) for
methods, source boundaries and tests. This update implements the analytics scope,
not the attached broader vehicle-profile/service-centre/RUL roadmap. Existing
RUL/risk restrictions and model metrics remain unchanged. Public deployment
still requires separate approval.

## Future deployment recommendation — not deployed


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

See the [showcase audit and unposted LinkedIn research summary](../reports/13_Showcase_Audit.md)
for current test evidence, release blockers, Git state and exact changed files.

## Meet the Developers


Two developers, complementary strengths, and a shared interest in building intelligent solutions.

### [Aarya Waghmare](https://github.com/AaryaWaghmare7)

**EVision role:** Software Engineering & Backend Development

- **Academic background:** B.Tech in Computer Science and Engineering
- **Institution:** Jawaharlal Nehru Engineering College, MGM University
- **Academic period:** 2024–2028

Computer Science and Engineering undergraduate focused on AI/ML, data analysis, and applied software development. Experienced in building machine-learning pipelines, image-forensics applications, and deployable web applications.

**EVision contributions:**

- Backend architecture and Python/FastAPI development.
- Implementation of application logic and prediction workflows.
- Machine-learning model integration.
- API endpoint development, input validation, and error handling.
- Frontend/backend integration.
- Testing, debugging, and application reliability improvements.
- GitHub development workflow, repository management, and version control.

**Technical interests:** Machine learning fundamentals, backend engineering, applied AI, software development, and building practical technology solutions.

**Wider academic & project experience:** SecureLens, an image-authenticity and fraud-detection application, and research into image compression and information loss.

### [Tejaswini Barhate](https://github.com/tejaswini-barhate)

**EVision role:** Data Analysis & Machine Learning

- **Academic background:** B.Tech in Artificial Intelligence and Data Science
- **Additional academic qualification:** Multidisciplinary Minor in Business Analysis
- **Institution:** Jawaharlal Nehru Engineering College, MGM University
- **Academic period:** 2024–2028

Artificial Intelligence and Data Science undergraduate contributing to the data analysis and machine learning aspects of EVision. Focused on understanding electric vehicle datasets, identifying relevant features, and preparing data for predictive maintenance applications. Interested in applying analytical and machine learning techniques to improve data-driven fault detection and vehicle health monitoring.

**EVision contributions:**

- Research and selection of the EV predictive maintenance dataset.
- Exploratory Data Analysis (EDA) to understand data patterns and distributions.
- Feature exploration and identification of relevant vehicle parameters.
- Data cleaning, transformation, and preprocessing.
- Preparation and inspection of datasets for machine learning experiments.
- Supporting fault diagnosis and predictive maintenance model development.

**Technical interests:** Machine learning, data analytics, predictive maintenance, electric vehicle technology, and business intelligence.

**Wider academic & project experience:** Experience with Python, SQL, Excel-based data analysis, Power BI, and database management.
