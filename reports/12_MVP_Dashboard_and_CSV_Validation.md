# Localhost Dashboard and CSV Repair Validation

Reviewed locally on 2026-10-09. No production-model retraining, tuning, metric
changes, commit or push was performed. Experiment 1 and Experiment 2 remain intact.

## Reproduced CSV failure and its cause

The running application was tested before edits, using real temporary CSV files
and multipart requests to `POST /analyze-csv` on port 8010:

| File content | Before repair | After repair |
| --- | --- | --- |
| Seven exact sensor headers, four supplied demo rows, no target | HTTP 200, predicted codes 0/1/2/3 | HTTP 200, same predictions |
| Same sensors plus `Fault Label` and `Note` | HTTP 422 | HTTP 200, extras reported and ignored |
| Surrounding whitespace in sensor headers | HTTP 422 | HTTP 200, header matching trims whitespace |

The rejected requests returned: `CSV must have exactly the seven unique
Experiment 2 feature headers`. Inspection found a strict header-length/exact-set
check in `src/api/telemetry.py`: it required exactly seven **untrimmed** header
names. This reproduced the extra-column/whitespace import failures. The endpoint,
POST method, existing FormData, multipart handling and installed dependency were
working; a valid seven-column upload already passed through the saved pipeline.
UTF-8 BOM support already existed and has been preserved.

The user's original failing CSV was requested but was not attached. Therefore
the cause of that specific file's failure cannot be established conclusively;
the findings above describe observed, reproducible failures, not a guessed
diagnosis of an unseen file.

## Backend repair and validation policy

- Decode UTF-8/UTF-8 BOM; trim surrounding header whitespace for matching only.
- Require all seven original sensor names; list every missing sensor in errors.
- Reject blank headers and duplicates, including duplicates after trimming.
- Accept extra columns as ignored metadata. Report them in `ignored_columns`;
  even a deliberately incorrect `Fault Label` cannot affect predictions.
- Document header-only normalization in `normalized_headers`. No uploaded file
  is rewritten, retained by the application, or used to replace a model.
- Check each record's width against the entire header, including ignored columns.
- Require every sensor cell to be finite, numeric and normalized to [0,1], with
  the existing 1e-12 boundary tolerance unchanged. Errors name the line and feature.
- Reject empty/header-only files, blank rows/cells, malformed quoting, nonnumeric,
  NaN/Infinity, out-of-range values and unsupported encoding. No rows are dropped,
  values invented, ranges clipped or physical readings rescaled.
- Construct a pandas DataFrame with **only** the seven sensors in canonical
  order; run existing shared validation and saved-pipeline inference.
- Validate the entire CSV before any prediction. Keep 5 MiB / 20,000-row limits,
  the 1,000 returned-prediction cap and whole-file summaries. Invalid data is 422,
  limit failures 413 and model unavailability 503.
- Declare `File(...)` explicitly for clarity; inferred multipart support was
  already working and was not the reproduced failure's cause.
- Manual JSON remains strict: exactly the seven original sensor aliases, with
  extras, strings, booleans, nulls, nonfinite and invalid-range values rejected.

Ignored timestamp/ID columns do not establish chronology, vehicle identity,
temporal analysis or cross-vehicle generalization. CSV rows remain independent
observations in uploaded file order.

## Dashboard changes

The existing plain HTML/CSS/JavaScript architecture is retained; no frameworks,
font downloads or additional packages were installed. The engineering workspace
uses restrained navy/green styling, system-font fallbacks, consistent cards,
subtle borders, readable helper text and responsive grids. Model Online is based
on `/health`, not a static claim. The normalized-input warning stays visible.

Four supplied sample buttons fill the seven fields only; Analyze Vehicle is an
explicit action. The result panel starts with Awaiting vehicle analysis, then
shows the predicted class and **Model Confidence** to one decimal place. It is
not called a future-failure probability or safety rating. The system panel labels
99.42% accuracy / 99.21% macro F1 as **Experiment 2 test-set performance**.

Telemetry UI shows the filename, processing state, named validation errors,
counts/percentages, CSS distribution bars, dominant abnormal classes including
ties, mean confidence and bounded row predictions. Additional columns and header
trimming are disclosed. The tiny downloadable CSV contains the four user-supplied
demo observations, not the raw dataset or a new labeled evaluation set. Its
historical static URL is preserved; download name is `sample_normalized_telemetry.csv`.

Both forms guard duplicate submissions, disable controls while processing,
recover after success/error and clear stale results after edits. CSV uses FormData
without manually specifying Content-Type/boundary. Network and non-JSON errors
are human-readable. Uploaded text is rendered with textContent, not HTML.

During development, a browser retained old CSS/JS while the new HTML was already
loaded. Versioned asset URLs now request the matching redesign assets. This was
a development refresh issue, not evidence of the original CSV failure's cause.

RUL says Coming in next module, requires a separate degradation/run-to-failure
model and remains disabled. Final risk scoring is disabled with no numerical
placeholder. Prototype limitations remain visible.

## Real-file HTTP checks

`scripts/smoke_test_localhost.py` opens actual temporary demo CSVs for multipart
upload and verifies each file's SHA-256 remains unchanged. All nine cases passed:

| Case | Result |
| --- | --- |
| Valid sensor CSV | HTTP 200 |
| Missing required column | HTTP 422, missing Voltage (V) identified |
| Nonnumeric sensor | HTTP 422, line 2 / Temperature (°C) identified |
| Empty file | HTTP 422, useful empty-file error |
| Extra harmless column | HTTP 200, Note ignored |
| Reordered sensors | HTTP 200, canonical order restored |
| Valid CSV without Fault Label | HTTP 200 |
| UTF-8 BOM plus whitespace headers | HTTP 200, seven trimmed headers documented |
| Deliberately wrong optional Fault Label | HTTP 200, label ignored; predictions unchanged |

Successful four-row files produce one prediction per class (25% each), with mean
class confidence 0.9975 (99.8% at UI display precision). The three abnormal
classes tie; no arbitrary winner is invented. These are application smoke checks,
not new model-evaluation metrics.

## Browser and regression checks

- All four sample buttons were checked in the actual browser: exact supplied
  values, no automatic prediction, expected codes/classes after Analyze.
  Confidence: Normal 100.0%, Motor Fault 100.0%, Inverter Fault 99.0%, Battery
  Fault 100.0%. Controls recovered after each request.
- Actual browser upload of a four-row file with extra label/Note columns returned
  a rendered four-class summary and ignored-column disclosure.
- An actual missing-column upload displayed `CSV is missing required sensor
  columns: Voltage (V)`, hid stale results and re-enabled upload controls.
- A clean browser session passed manual prediction and target-free CSV upload,
  with loaded styling and no console errors. The final review session on port
  8012 also passed prediction with no console errors.
- Laptop 1366×900 and mobile 390×844 layouts were checked. Document width equaled
  viewport width, without horizontal overflow. Temporary viewport overrides were
  reset after checking. Keyboard Tab focused Current with a visible 3px outline.
- A manual value of 2 was rejected by native range validation; stale results were
  cleared. Inputs have labels, range hints and keyboard focus styles.
- Health/model readiness, saved artifact loading, confidence output, unavailable
  model handling and existing Experiment 1/2/RUL contracts passed regression tests.
- The verification server's logs contained normal 200/expected validation 422
  responses with no traceback or unexpected 5xx during smoke checks.

## Test results and reproduction

Before edits: **103 passed**, two existing warnings. After changes:

```bash
python -m pytest -q
# 111 passed, 2 warnings

node --check src/api/static/app.js
node --test tests/frontend/test_dashboard.cjs
# Syntax valid; 7 frontend control-flow tests passed (Node is optional)

python scripts/smoke_test_nev_inference.py
# Saved model/classes/probabilities/column reordering passed; no fit

python scripts/smoke_test_localhost.py --base-url http://127.0.0.1:8012
# Health, four manual classes, assets and nine real-file CSV cases passed
```

The suite's existing Starlette/httpx deprecation and joblib physical-core
detection warnings remain nonblocking. No dependency upgrade was performed.
Tiny synthetic test fixtures are fitted only inside unit tests; the production
NEV artifact was never fitted or regenerated. The previous extra-header rejection
test was deliberately replaced by a stronger acceptance/non-leakage test to match
the user's new CSV policy; invalid-data protections remain tested.

Initial reproduction and browser upload checks used 8010. That server stopped
serving after an interruption. The existing verification server on **8012** now
provides the review preview; README's usual startup port remains 8010. Start any
available localhost port with `python -m uvicorn src.api.app:app --host 127.0.0.1
--port 8012` and match the smoke script's base URL.

## Preservation and remaining limitations

The saved pipeline SHA-256 is unchanged:
`010ff88628b7d58cb3aa1f8d2430b1087e3aef1936c37082edfde7270acbe987`.
Metadata and all four raw dataset hashes also match their pre-existing values.
Experiment reports 05–10, model results and training/preprocessing modules are
unchanged. Raw datasets remain Git-ignored; no upload fixtures entered the repo.

Inputs still require the dataset's normalized scale; original physical-unit
normalization parameters remain unknown. Performance is supplied-dataset-only,
not OEM/temporal/cross-vehicle evidence. The app is localhost research software,
not a production service or safety certification. Windows uses portable code and
documented startup commands, but execution on Tejaswini's laptop is still to be
confirmed. RUL/risk remain unimplemented. The original failing CSV still needs
to be supplied if its exact failure requires further investigation.

## Files changed and review boundary

Modified:

- `src/api/static/index.html`, `styles.css`, `app.js`, `synthetic_normalized_telemetry.csv`
- `src/api/app.py`, `schemas.py`, `telemetry.py`
- `tests/api/test_fault_api.py`
- `scripts/smoke_test_localhost.py`
- `README.md`, `PROJECT_PLAN.md`, `DECISIONS.md`

Created:

- `tests/frontend/test_dashboard.cjs`
- `reports/12_MVP_Dashboard_and_CSV_Validation.md`

Branch remains main. Changes are unstaged/uncommitted. Pre-existing unrelated
`distribution.java` and `separation.java` remain untouched. No commit, push,
dataset publication or additional model work is authorized in this review.

## 2026-10-15 — Black/charcoal frontend review

This follow-up supersedes the earlier light-palette description, not the
Experiment 1/2 results or earlier test evidence. The user clarified that RUL
data was downloaded on the collaborator's Windows laptop, but is unavailable
in this checkout. The current scope is frontend and fault/CSV validation only.

### Completed changes

- Matte black background, charcoal panels/inputs, silver primary actions,
  refined borders/shadows and readable secondary text. Green/amber accents
  describe fault-model status/labels, not calculated maintenance-risk levels.
- Reused existing HTML, CSS, JavaScript, SVG icons and responsive layout.
  No new files, frontend frameworks, fonts or dependencies were needed for
  this follow-up; model/API/prediction code is unchanged.
- Lightweight color/border/shadow hover transitions and a CSS loading indicator
  tied to the existing real request's `aria-busy` state. The reduced-motion
  media query disables animations/transitions. No arbitrary wait is added.
- Versioned asset references request the matching charcoal stylesheet/script.
- Bearing / Component RUL is explicitly pending a verified dataset; Maintenance
  Risk is pending validated evidence. Both controls remain disabled, with
  explanations and no numerical estimate, score, fabricated degradation curve
  or automatic recommendation.
- README, plan, decisions and the local RUL-data README record the clarified
  scope and blockers. One essential API/static regression test was added.

### Local verification

- `python -m pytest -q`: **112 passed**, two existing nonblocking warnings
  (Starlette/httpx deprecation and joblib physical-core detection).
- `node --test tests/frontend/test_dashboard.cjs`: **7 passed**, including
  duplicate-submit protection and success/error recovery. No npm install.
- HTTP smoke test on port 8012: health, saved artifact, four supplied diagnoses,
  static assets and all **nine real-file CSV cases passed**. Invalid uploads
  still name missing columns or the invalid line/feature.
- Actual browser keyboard checks: each sample fills without automatic inference;
  explicit Analyze returns Normal / Motor Fault / Inverter Fault / Battery
  Fault, with confidence 100.0% / 100.0% / 99.0% / 100.0%, respectively.
- Actual browser file chooser and multipart upload of the tiny downloadable
  sample: four rows analyzed, one row per class, **25% bars** for all four
  classes, mean model confidence displayed as 99.8%, no error. The chart is
  derived from real inference, not a fixed mock. These are smoke observations,
  not a new model evaluation dataset.
- Viewports **1366, 1024, 390 and 320px** checked: document width equals viewport
  width, no horizontal overflow; inputs change from three to two to one column.
  Browser reports dark color scheme and charcoal card backgrounds. Temporary
  viewport overrides are reset after review.
- No application JavaScript errors during successful diagnosis/upload checks.
  Existing model, metadata and recorded Experiment 2 metric hashes are unchanged.

### What remains blocked

No RUL model was selected or trained, so MAE/RMSE/R² and near-failure errors
are **not available**. Failure endpoints, run identity and labels cannot be
verified without the collaborator's dataset/documentation. No replacement data
was downloaded/extracted. Approximately 84 GiB was free during inspection;
dataset absence, not an assessed extraction-storage requirement, is the blocker.

No Low/Medium/High thresholds were invented. A risk engine requires defensible
maintenance limits and compatible component/unit/observation context; a NEV
fault-class probability cannot substitute for bearing RUL or a risk probability.
`src/evaluation/metrics.py` and all model functions/metrics are preserved.
Windows runtime testing remains to be confirmed on the collaborator's laptop.

Local review command (from the repository root):

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012
```

URL: `http://127.0.0.1:8012/`. Keep the server running while using localhost.
No commits, pushes, merges or deployments were performed. Existing unrelated
Java files and all earlier work are preserved. Stop for user review.

## Automotive reference redesign — 2026-10-26

The approved reference is now the visual direction: a cinematic red-lit EV hero,
black/charcoal panels, fire-red controls, a three-card analytics/performance/upload
overview and separate diagnostic, research and upcoming-module sections. The
existing project name is retained. Home, Diagnostics, Analytics, Models, Reports
and Documentation links have real destinations. No fictitious search, vehicle
IDs, recent-event timestamps, notifications, fleet counts or trend improvements
from the reference were copied. Real observations replace its fictional records.

Palette: Fire Red `#D10721`, Sleek Black `#1D1D1D`, Charcoal `#474549`, White
`#F5F5F5`, with restrained tonal variants. Inter is preferred when installed;
system fonts keep the page self-contained with no external font dependency.
CSS supports reduced motion, keyboard focus, responsive navigation, loading
spinners and short result/chart entrance transitions. The hero is decorative,
not evidence of a monitored vehicle or a supported manufacturer.

Original asset: `src/api/static/ev-hero.png` (about 1.8 MB), generated with the
built-in image-generation tool, not copied from the reference or a stock site.
Final generation prompt:

> Use case: ads-marketing. Asset type: original website hero background, wide 3:2
> landscape cinematic automotive render. Create an unbranded sleek black electric
> sport sedan seen from the rear three-quarter angle, facing left, placed on the
> RIGHT two-thirds of the image. Dark graphite studio garage with one dramatic
> diagonal fire-red illuminated wall panel, glowing red full-width tail light,
> restrained red rim lighting and subtle reflections on a dark floor. Premium
> photorealistic automotive technology aesthetic, near-black shadows, charcoal
> body, fire red #D10721 highlights, no blue. Keep the LEFT third nearly empty
> black negative space to allow website text overlay. No text, no logos, no badges,
> no watermark, no people, no interface. Original generic vehicle design, not an
> identifiable manufacturer. The supplied screenshot is mood/layout reference
> only, not an image to reproduce as a UI.

Explore Demo reuses the supplied Motor Fault observation, filling the form without
inference. Get Started opens Diagnostics. The four manual samples still return
Normal, Motor Fault, Inverter Fault and Battery Fault respectively (confidence
100%, 100%, 99%, 100%); these are functional checks, not new evaluation results.

The donut is gray with an em dash until valid telemetry is analyzed. Arc lengths
use exact whole-file prediction counts, while accessible legend buttons report
counts/percentages and filter returned rows. Download exports actual returned
predictions in original observation order, regardless of the display filter.
The first-1,000 detail cap remains explicit; no full-file export is implied for
larger files. New file selection, failed validation and new submissions clear
stale charts and disable downloads until a successful response. No uploaded text
is inserted as HTML. The CSV and manual backend contracts are unchanged.

The Model Performance card uses the saved evaluation: accuracy **99.42%**, macro
F1 **99.21%**, macro precision **99.23%**, macro recall **99.21%**. These are
Experiment 2 supplied-test-set scores, never live fleet statistics. The saved
pipeline and metadata hashes remain unchanged. No production model was fitted.

Browser checks covered all four manual classes, Explore Demo, valid CSV upload,
25% per-class donut for the four-row sample, Motor Fault filtering retaining
observation #2, and an actual four-row CSV download. All six navigation links
reached their intended sections, and a manual voltage value of 2 was blocked by
native form validation without showing a stale prediction. Missing Voltage (V) produced
a useful error, cleared the donut/table and disabled download while re-enabling
upload. Responsive checks at 1366, 1024, 390 and 320 CSS pixels showed no horizontal
document overflow. RUL and Risk remain disabled with their evidence requirements.

Validation: **118 pytest tests passed**, with the same two nonblocking warnings;
**11 Node control-flow tests passed**, including demo reuse, exact donut arcs,
stale-state cleanup and bounded downloads. The HTTP smoke checks passed all four
manual classes and nine real-file CSV cases. Existing experiments, raw datasets,
metrics and the trusted classifier are preserved. No new dependencies, training,
commit, push, merge or deployment were performed.

The reference is intentionally adapted into a longer, functional page: there
is no fictional fleet summary, severity classification, operational timeline,
RUL curve or live risk dial. The EV render is original rather than pixel-identical,
and typography uses the local system fallback when Inter is unavailable.

Files changed for this redesign (earlier uncommitted work was preserved):

- `src/api/static/index.html`, `styles.css`, `app.js`, and new `ev-hero.png`.
- `tests/api/test_fault_api.py`, `tests/frontend/test_dashboard.cjs`, and
  `scripts/smoke_test_localhost.py`.
- `README.md`, this report, and refreshed
  `reports/figures/showcase/dashboard.png` / `telemetry.png` screenshots.

Final browser console check returned no errors. Temporary responsive viewport
overrides were reset after review. The local server remains available on 8012.

## File-preview rendering diagnosis and motion verification — 2026-10-26

**Root cause:** the reported browser address was
`file:///…/src/api/static/index.html`, not the running FastAPI application.
The HTML intentionally imports `/static/styles.css` and `/static/app.js` and
calls same-origin API endpoints. Under `file:`, these absolute-root URLs point
at the filesystem root, not the repository or FastAPI; this explains the unstyled
document and default-size SVGs. There is one plain HTML entry point, no template
inheritance or frontend build output. The existing `/static` mount and its
project-relative filesystem directory are correct and were not replaced.

The localhost browser and HTTP checks established that the intended assets were
already present and served correctly. No localhost 404 or MIME-type mismatch was
found. Do not confuse a file-preview loading failure with a broken API mount.

Changes:

- Added a small, early inline `file:` guard that sends direct file previews to
  the documented default server, `http://127.0.0.1:8012/`. HTTP/HTTPS pages retain
  their own origin/port. The server must be running; custom ports must be opened
  explicitly. The redirect branch is unit-tested in isolation; functional browser
  testing uses the supported HTTP application, not a filesystem preview.
- Clarified the correct launch workflow in README. No personal paths or external
  hosting URLs were introduced.
- Added intrinsic 24×24 SVG dimensions and a zero-size symbol library, in addition
  to the existing CSS sizing. Browser-measured visible icon widths were 18–33px;
  the sprite width was zero. Bumped CSS/JS asset versions to `automotive-2`.
- Added Deep Black `#08090B`, keeping the approved red/black/charcoal palette.
- The prior hero was a static illustration. It now has a short image entrance,
  red-light sweep and headline reveal; scrolling drives bounded image translation
  and scale through a passive listener and one scheduled frame at a time.
  IntersectionObserver reveals real dashboard sections. This is image-layer
  parallax, not a 3D vehicle simulation or generated operational telemetry.
- CSS and native browser APIs provide motion without GSAP, ScrollTrigger, CDN
  scripts, package installs or a new framework. Reduced-motion CSS disables
  entrances/reveals; preference changes reset the scroll-motion amount to zero.

Actual localhost responses:

| Asset | Status | Content-Type |
| --- | --- | --- |
| `/static/styles.css?v=automotive-2` | 200 | `text/css` |
| `/static/app.js?v=automotive-2` | 200 | `text/javascript` |
| `/static/ev-hero.png` | 200 | `image/png` |
| `/static/synthetic_normalized_telemetry.csv` | 200 | `text/csv` |

Browser evidence: the loaded stylesheet contained 209 parsed rules, body background
computed to the intended deep-black gradient, and the hero image URL resolved to
the localhost `/static/` mount. Entrance translation was observed at about 53px,
settling to zero. Scrolling changed the image-layer matrix to approximately
`scale(1.058)` / `translate(-43.7px, 23.3px)`, and the analytics section became
visible at opacity 1. The console-error query returned an empty list.

Get Started navigated to Diagnostics. Explore Demo filled the genuine Motor Fault
sample without inference; Analyze returned Motor Fault. The other three samples
returned their expected classes. Browser CSV upload returned four observations,
one per class; Motor Fault filtering retained observation #2. The actual downloaded
CSV contained those four returned predictions, not fictional fleet records.

Regression result: **119 pytest tests passed**, with the same two nonblocking
warnings; **13 Node tests passed**. New tests cover file-mode routing, preserving
HTTP origins/ports, asset MIME types, intrinsic SVG dimensions, frame-bounded
motion and reduced-motion reset. All nine real-file HTTP CSV cases passed.
The model pipeline and metadata SHA-256 values remain unchanged. No model fitting,
dataset download, commit, push or deployment was performed.

The invalid-file browser check also passed after the motion changes: missing
Voltage (V) showed the named error, reset the donut to an em dash, hid stale
results, disabled download and re-enabled upload. Mobile document widths matched
390px and 320px viewports with all six navigation links retained.

Files changed in this repair: `src/api/static/index.html`, `styles.css`, `app.js`;
`tests/api/test_fault_api.py`; `tests/frontend/test_dashboard.cjs`;
`scripts/smoke_test_localhost.py`; `README.md`; this report; and the refreshed
`reports/figures/showcase/dashboard.png`. The existing hero image was reused.

## EVision cinematic implementation — 2026-10-26

### Branding and preserved boundaries

The application is now **EVision**, subtitled **AI-Powered EV Predictive
Maintenance**, with the tagline **Predict. Prevent. Drive Further.** Navbar,
hero, browser title, footer and API documentation metadata use the new brand.
The GitHub repository, directories, imports, endpoint paths, feature aliases,
prediction contracts and experiments retain their original names and behavior.
Only public branding strings changed in the backend; inference was not rewritten.

Before animation work, the actual localhost page was checked: dark CSS loaded,
the model was ready, Normal-sample inference passed and console errors were empty.
The existing file-preview guard remains; the supported app URL is HTTP localhost,
not a standalone HTML document. Current assets use version `evision-1`.

### Original asset and provenance

`src/api/static/evision-car.webp` is an original AI-generated transparent cutout,
created with the built-in image-generation tool from the project's existing
original `ev-hero.png`. It is not a stock/OEM asset, identifiable vehicle claim,
3D model or pasted reference screenshot. No third-party stock license was imported
or represented as cleared. Generated-asset provenance is recorded here; this does
not resolve the separate dataset/model redistribution limitations in report 13.
The old illustration remains preserved but is no longer downloaded by the page.

The generated PNG was converted to WebP with the already installed Pillow,
quality 90, retaining transparency. Runtime asset: **188,068 bytes (183.7 KiB)**,
**1774×887 RGBA**, alpha range 0–255. Only this local high-priority hero image is
loaded; there are no below-the-fold raster assets requiring lazy loading.
The generation prompt was:

> Use case: background-extraction. Asset type: transparent automotive website
> animation layer. Edit target: the existing original AI-generated black electric
> sedan in the supplied image. Extract ONLY the same complete unbranded car,
> including both visible wheels, windows, mirrors, black body and red tail-light/rim
> lighting. Preserve its rear three-quarter view facing left, realistic proportions,
> body design and photographic finish. Remove the entire garage, illuminated wall,
> floor, floor reflection and background. Reframe as a tightly composed wide
> landscape cutout with a small transparent margin around the entire car; do not
> cut off any part of the vehicle. The car should occupy almost the full width.
> Keep the car solid, with genuinely transparent alpha outside the silhouette.
> No text, logos, badges, extra cars, ground plane, invented scenery or watermark.
> This will be translated/scaled as a flat image, not falsely rotated in 3D.

### Three scenes and accessibility

1. CSS introduces the car from the right over 1.8 seconds with decelerating easing,
   an independent red-light sweep and coordinated branding/headline reveals.
2. Bounded scroll progress drives car translation/scale, lighting displacement,
   red accent lines and heading fade/translation. Movement reverses on upward
   scrolling. Normal document flow is retained: no pinning or scroll locking.
3. IntersectionObserver reveals individual dashboard panels with short staggered
   delays. Only real uploaded results populate/animate the chart and result cards.

Native browser APIs and CSS were retained instead of adding GSAP/ScrollTrigger,
Three.js, CDN dependencies or a build system. There is at most one pending render
frame; scroll listeners are removed while the hero is offscreen, reduced motion
is selected, or the viewport is compact. Page-exit cleanup disconnects observers,
media listeners, pending frames and control handlers; back/forward-cache restoration
reinitializes motion. Mobile views use a shorter entrance with no scroll parallax.
No persistent animation loop runs in the final application.

System reduced-motion preferences and the footer control share the same reduced
mode. Browser testing of the control confirmed `animation-name: none`, no car
transform, progress zero after scrolling, and dashboard cards at opacity 1.
The system-preference branch and listener cleanup are also unit-tested.

### Browser performance observation

A temporary, opt-in four-second requestAnimationFrame/PerformanceObserver probe
was read through the browser developer console while entrance and bidirectional
scrolling ran. It was then removed from application code. This is a short browser
runtime sample, not an exhaustive DevTools Performance trace or compositor/GPU
certification. The sample returned:

| Measurement | Observed result |
| --- | ---: |
| Frame intervals sampled | 234 |
| Median interval | 16.7 ms |
| 95th-percentile interval | 17.6 ms |
| Maximum interval | 83.7 ms |
| Long-task API supported | Yes |
| Reported main-thread tasks over 50 ms | 0 |

The median is approximately a 60 Hz callback cadence, but the maximum gap shows
that perfectly uniform frames cannot be claimed. The gap's cause was not
established; do not infer zero jank or guaranteed 60 FPS on all devices. Full
hardware/device profiling remains a release check. Asset size, normal document
flow, transform/opacity motion and offscreen cleanup limit the work performed.

### Functional validation

All six navigation destinations and Explore Demo were verified in the browser.
All four manual samples returned the expected classes with confidence
100%, 100%, 99%, 100%. CSV upload returned four observations, one per class;
Motor Fault filtering retained observation #2 without changing whole-file counts.
The saved model and evaluation metadata hashes remain unchanged. RUL and Risk
stay disabled pending separate scientific validation.

HTTP smoke checks passed all four manual classes and all nine real-file CSV cases.
CSS/JS/WebP/sample CSV returned HTTP 200 with `text/css`, `text/javascript`,
`image/webp`, `text/csv`, respectively. No dataset download, production fitting,
commit, push, merge or deployment was performed.

Final regression suite: **119 pytest tests passed**, with the same two existing
nonblocking warnings, and **15 Node frontend tests passed**. JavaScript syntax
and `git diff --check` passed. Responsive document widths matched 1366, 1024,
390 and 320 CSS pixels. Narrow-screen car framing was corrected to avoid clipping
the rear edge in its final position. Get Started opened all seven sensor inputs;
no horizontal document overflow was observed. The final console-error check was
empty. Temporary viewport overrides were reset after screenshot capture.

Files changed for EVision: `src/api/static/index.html`, `styles.css`, `app.js`;
new `src/api/static/evision-car.webp`; `src/api/app.py` (public branding strings
only); `tests/api/test_fault_api.py`; `tests/frontend/test_dashboard.cjs`;
`scripts/smoke_test_localhost.py`; `README.md`; this report; and refreshed
`reports/figures/showcase/dashboard.png` / `telemetry.png`. No existing model,
dataset, experiment report or repository directory was renamed or removed.

Local command: `python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012`.
Open `http://127.0.0.1:8012/`. RUL/risk scientific validation, full device/GPU
profiling and the pre-existing redistribution/license review remain outside
this local UI implementation. Stop for approval before any commit/push/deploy.

## Approved GitHub collaboration checkpoint — 2026-10-27

The user now authorizes verifying, committing and safely pushing EVision to
`origin/main` for Tejaswini's Windows testing. Earlier no-commit/no-push statements
describe their historical review scopes; no deployment or model retraining is
authorized. The newer remote README contribution is retained in Git history and
its research overview, Experiment 1 results, comparison, future-module roadmap
and limitations are reconciled with EVision and PowerShell startup instructions.

Checkpoint checks: 119 pytest tests and 15 frontend tests passed; saved-model and
HTTP smoke checks passed, including all four classes and nine real-file CSV cases.
The live browser confirmed styling, moving car, navigation, demo diagnosis and
CSV chart/filter behavior. `pip check` reported no broken requirements. The
standard Python 3.13 Windows x64 wheels for pandas 2.3.3, NumPy 2.3.5 and
scikit-learn 1.7.2, plus universal wheels for the other pinned direct runtime
packages, were verified against PyPI metadata without installing packages.
This is not a native Windows runtime test; that remains Tejaswini's next check.

No Mac interpreter is configured in shared VS Code settings. Runtime/model paths
are repository-relative. The already tracked trusted inference model remains
unchanged, and raw CSVs/RUL archives stay ignored. The tiny four-row frontend
sample CSV is an intentional demo asset, not a raw training/test dataset.
Unrelated `distribution.java` and `separation.java` remain local and excluded.
