# EVision reliability and security review — 2026-10-30

## Initial audit, before implementation

- Branch `main`, tracking `origin/main`; existing README modification and two
  unrelated Java files are preserved. No commit, push or deployment is approved.
- FastAPI serves local HTML/CSS/JS and WebP. No frontend build, external animation
  dependency, database service, authentication, user records or admin routes exist.
- The trusted, tracked Random Forest pipeline is loaded once, with feature-order,
  class and sklearn-version checks. Inputs are the original seven normalized
  sensors; class codes are 0 Normal, 1 Motor, 2 Inverter, 3 Battery.
- Baseline tests: **119 pytest passed**, **15 Node frontend tests passed**.
  `pip check` found no broken requirements. Two existing pytest warnings concern
  Starlette/httpx deprecation and joblib physical-core detection.
- All four supplied samples have regression tests. Explore Demo explicitly fills
  the Motor sample. Its predicted-class confidence can legitimately equal 1.0;
  that is not a real-world failure probability. No universal Motor fallback was
  found. The user's offending upload is not available, so its cause cannot be
  established from the description alone.
- CSV parsing already rejects invalid sensor rows before inference, orders inputs,
  ignores extra fields (including labels), caps bytes/rows and avoids persistence.
  It lacks an inspect-only schema report, delimiter support, explicit evaluation,
  multiple-file UI and supported database inspection.
- The saved artifact and prior supplied-test results are preserved. Reports 08–10
  document sharp class boundaries, no exact train/test overlap, unknown source/
  normalization/label-generation protocol and missing time/vehicle identifiers.
- No scientifically validated local RUL model or risk thresholds exist. These
  outputs must remain unavailable, not derived from classification confidence.
- Verified hardening gaps: no CSP, request-origin/Host defense, rate/concurrency
  controls, production access gate or disabled production docs. Existing request
  byte limits and three security headers are useful but not a complete boundary.

## Prioritized implementation plan

1. Preserve the artifact; reproduce its fixed supplied-test results without fitting.
2. Clarify demo/confidence semantics and add versioned, conservative guidance.
3. Reuse validation/inference for inspect-only uploads, explicit prediction versus
   user-attested evaluation, independent file reports and bounded read-only SQLite.
4. Add narrow security controls and negative tests; avoid introducing accounts,
   private storage, remote database connections or model retraining.
5. Run full tests, live API and browser flows; document genuine results, unresolved
   provenance/calibration/RUL/risk limits and security verification boundaries.

## Findings and implementation

### The 100% Motor result

No incorrect class encoding, probability alignment, model-order change or Motor
fallback was reproduced. The saved model and live API distinguish all four
supplied examples: Normal 1.00, Motor 1.00, Inverter 0.99, Battery 1.00 confidence.
Explore Demo intentionally uses `loadSample(1)`. The UI now states that explicitly.
A CSV containing only Motor-like observations can legitimately have 100% Motor
**prediction count**; this is distinct from an individual class confidence and
neither is population fault prevalence. Invalid inputs are rejected, not filled
with defaults. The original offending CSV/values were not provided; its specific
cause remains unverified and was requested from the user.

**Follow-up: exact case supplied and reproduced.** The initial missing-input
limitation above is now resolved for the seven-value case below. No further
request for these values is needed.

| Original model feature (UI label) | Exact supplied value | Preprocessor output |
| --- | ---: | ---: |
| Voltage (V) (Voltage) | 0.80 | 0.8 |
| Current (A) (Current) | 0.597 | 0.597 |
| Motor Speed (RPM) (Motor Speed) | 0.073 | 0.073 |
| Temperature (°C) (Temperature) | 0.10 | 0.1 |
| Vibration (g) (Vibration) | 0.92 | 0.92 |
| Ambient Temp (°C) (Ambient Temp) | 0.28 | 0.28 |
| Humidity (%) (Humidity) | 0.12 | 0.12 |

The form's `name` attributes are the original aliases in the first column;
JavaScript converts number-field strings to JSON numbers in that order.
Pydantic validates every alias, type and normalized range. The API constructs
a seven-column DataFrame in `NEV_FEATURES` order; shared inference restores
that order again. The saved ColumnTransformer selects these exact names and
uses its **already fitted median SimpleImputer**. This complete observation
requires no imputation, and the transformed array equals the input exactly.
There is no scaling, refitting, physical-unit conversion or target input.

The saved pipeline, HTTP API and actual rendered browser all return:

```text
predicted_class = 1
predicted_fault = Motor Fault
classes_ / class mapping = [0 Normal, 1 Motor Fault, 2 Inverter Fault, 3 Battery Fault]
predict_proba = [0.0, 1.0, 0.0, 0.0]
model_confidence = 1.0
display = 100.0%
```

All **200 trees** return a leaf Motor class probability of 1 for this input.
Thus the exact case is not a rounding, label-mapping, wrong-feature-order or
stale-response defect. The browser uses `100 * model_confidence` with one decimal
place; it correctly displays 100.0 here. No diagnosis or probability is changed.
The case differs from the Motor demo in voltage, temperature, vibration, ambient
temperature and humidity; regression tests preserve those differences. It is
not an exact seven-feature row in either supplied training or testing file.

Training-only observed Vibration distributions (normalized, not physical g):

| Label | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Normal | 0.000000 | 0.184709 | 0.363595 |
| Motor | 0.454594 | 0.714350 | 0.999149 |
| Inverter | 0.000129 | 0.236300 | 0.454515 |
| Battery | 0.000029 | 0.131616 | 0.272685 |

All **242** actual training observations with Vibration ≥ 0.90 have label 1.
The supplied value 0.92 is in this unusually class-separated region. This is
descriptive evidence of dataset association, not a newly validated engineering
threshold, proof that vibration caused a fault, or evidence of label-generation
provenance. No synthetic counterfactual readings or physical conversions were
invented. All four unchanged demo samples still predict their respective classes.

The underlying reliability issue is **extremely confident predictions on a
simple, sharply separated supplied dataset with unverified real-world meaning**,
not a frontend percentage needing cosmetic correction. Dataset calibration
diagnostics below do not establish individual real-world certainty. The UI now
shows the full vector and says unanimous-tree confidence is not diagnostic
certainty. Tests cover direct transformation, tree probabilities, exact API
mapping and frontend serialization/display. No model change is justified by this
case alone. Compatibility with raw physical readings remains unknowable without
the original normalization protocol; no replacement transform is introduced.

### Model reliability — no improvement claim

Full/train/test files retain 11,000 / 7,700 / 3,300 rows with eight columns.
The supplied split still reconstructs the full multiset, with zero exact
feature overlap. Raw file SHA-256 checks match report 08. The approved model
SHA-256 remains `010ff88628b7d58cb3aa1f8d2430b1087e3aef1936c37082edfde7270acbe987`.
The canonical artifact's hash is now checked before joblib deserialization;
the application never accepts an uploaded artifact or fits during inference.

| Supplied-test metric | Recorded before | Reproduced unchanged |
| --- | ---: | ---: |
| Accuracy | 0.9942424242 | 0.9942424242 |
| Macro precision | 0.9922778286 | 0.9922778286 |
| Macro recall | 0.9920833333 | 0.9920833333 |
| Macro F1 | 0.9920817442 | 0.9920817442 |
| Macro OvR ROC-AUC | 0.9996980710 | 0.9996980710 |

No tuning, calibration fitting, resampling or new candidate fitting was done.
Historical comparison remains LR 0.980909 accuracy / 0.975324 macro F1;
RF 0.994242 / 0.992082; HGB 0.993333 / 0.990832. Training-only CV selected RF
before the historical final test. A fixed always-Normal reference gives
0.454545 test accuracy. Historical metrics files were not overwritten.

Confusion matrix (actual rows / predicted columns, order 0–3):

```text
1500    0    0    0
   0  600    0    0
   0    0  599    1
   0    0   18  582
```

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Normal | 1.000000 | 1.000000 | 1.000000 | 1500 |
| Motor | 1.000000 | 1.000000 | 1.000000 | 600 |
| Inverter | 0.970827 | 0.998333 | 0.984388 | 600 |
| Battery | 0.998285 | 0.970000 | 0.983939 | 600 |

New **descriptive diagnostics of the unchanged model**, through the extended
`src/evaluation/metrics.py` workflow: macro OvR Average Precision 0.9986259590;
multiclass Brier (sum over four classes, no division by two) 0.0100853333;
log loss 0.0219595735. Fixed ten-bin top-label reliability:

| Confidence bin | Count | Mean confidence | Observed accuracy |
| --- | ---: | ---: | ---: |
| [0.5,0.6) | 4 | 0.581250 | 0.250000 |
| [0.6,0.7) | 14 | 0.665000 | 0.857143 |
| [0.7,0.8) | 18 | 0.746944 | 0.888889 |
| [0.8,0.9) | 43 | 0.851744 | 0.860465 |
| [0.9,1.0] | 3221 | 0.994437 | 0.998137 |

Other bins are empty, not zero accuracy. Sparse low-confidence bins cannot
support confident calibration conclusions. Brier/log loss assess more than
calibration alone; these diagnostics do not justify interpreting confidence as
fleet fault/failure probability. See [sklearn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

Unknown source/license, normalization and label-generation protocols remain
blockers to wider scientific claims. Sharp boundaries (historical depth-3 tree
CV macro F1 about 0.979) could reflect simplified/simulated rules; that origin
is **not established**. No timestamps/vehicle IDs means no temporal/cross-vehicle
claim. No new external dataset was downloaded and no outlier was deleted.

### Ingestion and interpretation

- Existing comma-CSV prediction API preserved; new `/analyze-dataset` supports
  inspect/predict/evaluate with explicit scale/label confirmation and column
  mapping. It never guesses unit conversions or uses labels as inputs.
- UTF-8/BOM, comma/semicolon/tab/pipe CSV, whitespace and reordering supported.
  Limits: 5 MiB, 20,000 rows, 100 columns, 128-character names. Quality reports
  distinguish missing/nonnumeric/nonfinite/out-of-range cells, observation
  numbers (and ending physical CSV lines in inspection), duplicates and ranges.
  Rejected uploads return no partial predictions or metrics.
- Multiple UI files are processed separately/sequentially, maximum five; a failed
  file does not erase another report. There is no merge or private history.
- SQLite is identified by bytes, loaded into memory and read query-only. Users
  select a listed ordinary table; no uploaded SQL, views, virtual/generated
  columns, extension calls or remote credentials. Identifiers are allowlisted
  and quoted; LIMIT is bound. Authorizer, schema restrictions, byte/row/column/
  expression/SQL-size limits and a two-second progress budget reduce exposure.
  No original database is modified. Python without `deserialize` fails safely
  and requests a CSV export. This follows [SQLite's untrusted database guidance](https://www.sqlite.org/security.html),
  but is not a separate-process sandbox or proof against native-library bugs.
- Uploaded-data evaluation reports per-class metrics, fixed class-order matrix,
  applicable AUC/AP and probability diagnostics. Missing classes produce null
  AUC/AP, not fabricated scores. No accuracy without labels. Exact original
  training/full fingerprints are blocked; reformatted/subset/related data can
  still overlap training. **Every uploaded score is marked non-independent**,
  with user-attested label semantics, not new generalization evidence.
- Versioned JSON knowledge gives normalized sensor context, possible causes,
  technician checks and cautions, not causal attribution or physical thresholds.
  General safety source: [NHTSA](https://www.nhtsa.gov/vehicle-safety/electric-and-hybrid-vehicles).
  Do not service high-voltage systems without appropriate qualification.
- Inspection workflow rule `inspection-guidance-1`: class 0 → `routine_review`;
  classes 1/2/3 → `technician_review_requested`. Inputs, rule version, missing
  evidence and absence of a horizon are explicit. This is not validated urgency,
  Low/Medium/High risk or probability. Confidence is not an input to that rule.
- `data/rul/` and `models/rul/` contain README scaffolding only. No verified
  run-to-failure endpoints, units/IDs or local model exist. RUL/future-event risk
  remain unavailable in the API guidance and disabled in the UI. Appropriate
  component-life data, OEM limits and separately approved validation are required.

### Security findings and fixes

- New CSP, same-origin/Host checks, request content-type validation and no-store
  responses supplement existing body caps, nosniff, frame and referrer headers.
  Application scripts are same-origin with an exact inline preview-guard hash;
  dynamic chart/animation style attributes require the documented inline-style
  exception. Swagger's separate CDN/hash allowance exists only in local mode.
- Optional environment Bearer token uses constant-time comparison; production
  refuses missing/short tokens or absent explicit hosts, disables docs/OpenAPI
  and adds HSTS only for HTTPS. No secret is committed or logged in responses.
  There are no roles, user records, admin endpoints, passwords or cookie sessions
  to migrate. This is not a complete multi-user authentication system: no token
  expiry, per-user revocation or password enrollment. Do not invent accounts.
- Single-process client limits: 60 manual/minute, 20 uploads/minute; two active
  uploads; actual body caps including chunked requests. Wrong Origin/Host denied;
  no arbitrary CORS reflection. Rate limits are not a distributed production
  boundary; use an approved gateway/shared limiter if deploying multiple workers.
- Static path/traversal checks do not expose `.env`, source or model files.
  Uploaded names never become disk paths. Browser rendering uses text nodes,
  not uploaded HTML. Predictions exports contain controlled model fields only.
- `pip check` passes. Initial OSV check of direct pins found no matches, but
  walking the 26-package installed runtime closure found AnyIO 4.10.0 advisories
  [stderr blocking](https://github.com/advisories/GHSA-5p39-cfhj-2xmp),
  [IDNA/TLS behavior](https://github.com/advisories/GHSA-82r6-8w77-94w6) and
  idna 3.11 [resource consumption](https://github.com/advisories/GHSA-65pc-fj4g-8rjx).
  Targeted AnyIO 4.14.2 / idna 3.15 pins were installed; the resulting 25-package
  runtime closure had **no OSV advisory matches**. No ML dependency changed.
  OSV coverage is not exhaustive; optional research/notebook dependencies and
  native Python/SQLite/OS libraries were not given a complete vulnerability audit.
- No standalone gitleaks/detect-secrets/pip-audit executable was available. A
  heuristic scan of all 402 historical Git blobs found no private-key/header,
  GitHub token, AWS access-key or long OpenAI-key pattern matches. No credential
  values were printed. This does not cover all credential forms or external
  clones/remotes. No history rewrite or credential rotation was attempted.

## Verification and scope boundary

Baseline: 119 pytest / 15 frontend tests. Final automated and browser results
are recorded below after the final verification pass. Tiny synthetic unit-test
models may fit in tests; the production classifier is never fitted or replaced.
No large downloads, external-system attacks, migrations, commits, pushes or
deployment are part of this task. Existing README edits and unrelated untracked
files are retained. There is no production frontend build for this buildless app.

### Final verification results

- `python -m pytest -q`: **152 passed, zero failed, zero skipped**, 6.74 seconds
  on this checkout. Two nonblocking warnings remain: Starlette/httpx TestClient
  deprecation and joblib's physical-core fallback. Optional raw-dependent checks
  may skip in a clone without datasets; inference does not need those CSVs.
- Node built-in runner: **19 passed**, including exact-case form serialization,
  explicit evaluation confirmation, independent files/error recovery, text-only
  rendering, stale-state clearing, download and reversible/reduced motion checks.
- `node --check src/api/static/app.js`, Python compile checks, `pip check` and
  `git diff --check` passed. No separate lint/type-check tool was configured or
  installed; this is not a claimed lint/type-check run.
- Real HTTP smoke: all four demos and the reported exact case, reordered JSON,
  invalid/missing fields, current CSS/JS/WebP/sample CSV HTTP 200 with correct
  MIME types; nine CSV cases passed (valid, missing sensor, nonnumeric, empty,
  harmless extra, reordered, no target, BOM/whitespace, ignored wrong target).
- Real browser: entered all seven reported values, observed Motor 100.0% and
  the complete [0,1,0,0] vector. All four demo diagnoses/guidance work. Original
  four-row CSV displays four 25% groups, not a universal Motor result.
- Browser upload inspection ran without inference. Two-file evaluation kept
  separate reports: four labeled demo observations yielded expected same-sample
  scores with an explicit non-independent warning; an incompatible physical-like
  CSV was rejected with all missing canonical sensor names. These demo labels
  are regression fixtures, not a new scientific test set.
- Mobile: 390px viewport/document widths matched, dataset form usable without
  horizontal page overflow; reduced motion disabled the car animation. Existing
  dark styling/local car and forward/reverse scroll transforms were verified.
  No console errors/warnings observed in the final app checks.
- A blank Swagger page was found during browser testing: the generated OAuth
  redirect initializer differed from the CSP hash input. Corrected the exact
  hash and added an HTML-hash regression test; real browser now renders all routes
  and schemas. Production still disables docs/OpenAPI. Swagger requires its CDN;
  the EVision dashboard itself does not.
- SQLite ordinary-table inference, view/generated/virtual/injection rejection,
  unchanged database bytes, wrong labels, rate/concurrency limits, hostile
  Origin/Host, static traversal, production token failures, disabled docs and
  HTTPS-only HSTS are covered by actual API tests. No native Windows test,
  public HTTPS deployment, external penetration test or distributed limiter was
  run. OSV and heuristic scans do not certify complete security.

### Important files changed

- `src/api/datasets.py`, `security.py`, `app.py`, `schemas.py`, `telemetry.py`:
  ingestion/evaluation, access/resource controls, API integration and shared
  one-pass inference summary.
- `src/prediction/guidance.py`, `fault_guidance.json`: versioned educational
  guidance and inspection workflow; no validated risk/RUL output.
- `src/storage/model_registry.py`: canonical pre-deserialization checksum.
- `src/evaluation/metrics.py`: additive probability diagnostics, existing metrics
  helpers preserved.
- `src/api/static/index.html`, `app.js`, `styles.css`: explicit workflows,
  full probability display/guidance, token field and mobile form styling;
  car asset/animation design preserved.
- `tests/api/test_dataset_security.py`, `tests/prediction/test_saved_nev_model.py`,
  `tests/evaluation/test_metrics.py`, `tests/frontend/test_dashboard.cjs` and
  `scripts/smoke_test_localhost.py`: regression and security verification.
- `.gitignore`, `.env.example`, `requirements-api.txt`, `README.md`, this report:
  placeholders, targeted security pins, setup and honest limitations.

No Experiment 1/2 historical report, metrics JSON, training code, raw CSV or
production model bytes were altered. Git HEAD remains `1868ea5`; no staging,
commit, push, merge or deployment occurred. Existing unrelated Java files and
the untracked `to/` directory are untouched. The working server uses
`python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012` at
`http://127.0.0.1:8012/`.

### Remaining decisions before broader release

Resolve NEV provenance/license, original normalization and label timing; obtain
independent vehicle/time-aware validation and suitable calibration evidence.
Supply verified component-life data before RUL, and compatible limits/horizons
before real risk scoring. A multi-user/private-record service needs a separately
reviewed identity/authorization design, not this shared token. Public release
also requires TLS/proxy/ingress controls, isolated processing review for hostile
uploads, native-library/optional-dependency auditing and native Windows checks.
Do not treat the passing tests or high same-dataset scores as approval to deploy.
