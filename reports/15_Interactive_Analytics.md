# EVision interactive analytics — 2026-10-31

## Scope and preservation

Implements the eight analytics requirements in the user's message, using report
09 and the code/data underlying its figures. The attached platform-wide brief is
broader: RUL, vehicle catalogs and service-provider integrations are not claimed
complete by this analytics change. No new model, dataset download, dependency,
commit, push or deployment is performed. Prior pending security/reliability edits
and unrelated user files are retained. Experiment 1/2 results, models and original
figures are unchanged.

Baseline: **152 pytest tests passed**. The saved Random Forest, exact reported
Motor regression case and original four samples remain covered. Current model
test performance is unchanged: accuracy 0.994242, macro F1 0.992082, macro OvR
ROC-AUC 0.999698; analytics does not establish better model performance.

## Data sources, not screenshot substitution

The dashboard uses native SVG and real numeric summaries, not pasted screenshots.

| Source shown in the selector | Provenance | Permitted interpretation |
| --- | --- | --- |
| ORIGINAL NEV training | Hash-verified 7,700 supplied training observations | Class distribution/boxplots, joint-bin scatter, associations and training comparison reference |
| ORIGINAL NEV full EDA | Hash-verified 11,000 supplied observations | Descriptive histograms/correlation and published aggregate class counts, not blind holdout evaluation |
| UPLOAD with filename/file sequence | Current request's actual mapped observations | Its own summaries, with label basis and normalization confirmation shown |
| Original model evidence | Saved `signal.json` and `model_comparison.json` | Historical training MI/validation permutation/final-model impurity importance; never upload-specific causal attribution |

Report 09's training counts are 3,500 Normal / 1,400 Motor / 1,400 Inverter /
1,400 Battery. Full counts are 5,000 / 2,000 / 2,000 / 2,000. Proportions are
45.4545% Normal and 18.1818% each fault in both sources. Full and training
correlations are deliberately separate, so their displayed rounded cells can
differ. Full voltage/temperature Pearson r is approximately −0.498132.

`scripts/build_analytics_reference.py` checks all three raw-file fingerprints
against existing model metadata and verifies the supplied split. It computes
aggregate descriptive data and copies the historical associations/importance;
it never loads/fits a training candidate or calls the original experiment runner.
The small generated JSON contains no exact row vectors. Original scatter is a
20×20 joint-bin count representation per class/feature pair, with that limitation
shown in the UI. Binned centers are not invented individual observations.

No local raw CSV is needed to view the prepared snapshot. No source/license
claim beyond the existing reports is added; NEV provenance and normalization
parameters remain unresolved. Raw CSVs stay ignored.

## Implemented charts and interactions

1. Four-class counts and percentages with source and label-basis captions.
   Whole-file counts stay unchanged when filtering sensor charts.
2. Histograms for all seven named features. Normalized-contract data uses 35
   fixed bins plus under/overflow; out-of-range raw-scale uploads additionally
   get actual-range descriptive histograms without conversion or inference.
3. Sensor-only Pearson heatmap. Three finite paired observations are required;
   constant/insufficient cells are null, not zero. Tooltips include pair counts.
   Numeric fault codes are not treated as an ordinal correlated sensor.
4. Class boxplots with Q1/median/Q3 and **observed min/max whiskers**. This is
   explicitly labeled, rather than claiming the original static figure's
   Tukey-whisker/flier rendering. No outlier rows are deleted.
5. Feature-selectable scatter with class colors, point/bin tooltips and finite
   min/max range controls. Reset restores data bounds. Upload points retain
   original observation numbers; original reference marks represent bin counts.
6. Eta-squared class-mean association from the selected upload's labels/grouping;
   null when unsupported. MI is shown only from the original saved training
   analysis. Saved validation permutation importance retains its recorded mean
   and SD; final-model impurity importance is separately labeled. None is causal.
7. Upload/training range/median comparison, histogram total variation and a
   selected-sensor proportion overlay. The class filter is disabled for this
   whole-upload comparison and historical evidence, avoiding false per-class
   importance or comparison interpretations.
8. Missing/nonnumeric/nonfinite/range and descriptive shape-review warnings.
   Unknown normalization remains visible even if the numeric ranges look alike.

Original reference versus upload selection, source/class filters, keyboard
focus tooltips, responsive chart containers, explicit zoom and unavailable states
are implemented. The premium dark/red design and cinematic car remain intact.
Only the active chart view is mounted. There is no new chart dependency/CDN,
unbounded point rendering or animation loop.

## Upload integration and scientific boundaries

Both existing routes feed the same analytics functions:

- `/analyze-csv`: strict compatible numerical prediction workflow; grouping
  explicitly says `model_predictions`, not confirmed failures.
- `/analyze-dataset`: inspect/predict/evaluate workflow from the prior update.
  Inspect-only charts do not require model inference. Explicit mapping applies
  before charting. In evaluation, class profiles use valid user-confirmed labels;
  predictions remain separate. Invalid/incomplete data does not become eligible
  for inference merely because some cells can be charted.
- `/analytics/reference`: original aggregate snapshot only. No endpoint stores
  or retrieves private uploads; no dataset is merged with another.

Every histogram/box/correlation uses all available request observations, not
the capped returned prediction list. Nonfinite/nonnumeric chart cells are
excluded featurewise/pairwise with exact counts. No imputation, scaling or row
deletion is applied to make chart input eligible for prediction.

Upload scatter is bounded at **1,000 points**, deterministically sampled with
seed 42 and equal class budgets for display. It is not a representative class
mix estimate: whole-source counts supply proportions. This never changes model
inputs or any train/test membership. Samples may overlap visually; their original
row numbers and exact plotted values remain in tooltips/JSON. Aggregate statistics
are not sampled. Original reference never exports raw training vectors.

Data quality and distributions are not a vehicle safety certification. Values
retain the uploaded numeric scale; physical-unit names do not recover unknown
normalization. Correlation/association/importance do not establish causation,
calibration, confirmed component damage or temporal/cross-vehicle/OEM validity.
No timestamps, vehicles, fleet records, degradation curves or maintenance history
are invented. RUL/future-event risk remain unavailable.

## Range and shape review methodology

- Range flag: finite uploaded value below the original **training** observed
  minimum or above its maximum, allowing the existing 1e-12 roundoff tolerance.
  These are observed dataset bounds, not manufacturer operating limits.
- Shared comparison histogram: 35 fixed [0,1] bins plus below/above-range bins.
  Relative frequencies use each file's finite count, avoiding sample-size effects.
- Distance: total variation, `0.5 × sum(abs(upload_frequency − train_frequency))`,
  between these shared histograms (0–1). It is a descriptive effect measure.
- Shape review is triggered when distance >0.25 and at least 30 finite values
  exist for that sensor. Constants in `src/eda/dashboard.py` make this configurable.
  **This is an unvalidated review heuristic, not a significance test or calibrated
  OOD decision.** Smaller samples show the distance but no shape-alert conclusion.
- Class mix, normalization mismatch, coarse bins or domain differences may explain
  a flag. Similar marginals do not establish joint-distribution compatibility or
  exclude unknown distribution shift. No threshold changes inference, clips values
  or produces maintenance urgency/failure probability.

## Exports, privacy and deployment boundary

JSON retains the source, label basis, complete aggregate statistics, comparisons,
methods and bounded scatter. CSV includes source/scope/class-filter context and
seven-sensor summary values. The standalone HTML report snapshots the selected
view and contains numeric JSON in a details section; it is static, script-free
and printable to PDF using the user's browser. Native binary PDF generation is
not claimed. Exports are local browser downloads, not server-side report storage.

Untrusted filenames/labels are rendered as text, not HTML. Reports serialize
escaped DOM; standalone CSP forbids script/network execution. CSV cells are quoted
and source names have a fixed textual prefix; canonical feature names/numbers
are used, not arbitrary uploaded spreadsheet formulas. Existing auth, CORS,
request limits, model checksum and raw-data exclusions remain intact.

To reproduce the reference with the original local files:

```bash
python scripts/build_analytics_reference.py
```

Then restart the same FastAPI service. Missing reference data produces an honest
503/unavailable state, not fabricated training statistics. The snapshot should be
included in a later **approved** commit; no Git publication is performed now.

## Verification

Final automated/browser evidence is appended after the last verification pass.

### Final results

- **167 pytest tests passed**, no failures or skips on this local checkout
  (7.06 seconds in the final run). Two existing warnings remain: Starlette/httpx
  TestClient deprecation and joblib's physical-core detection fallback.
- **28 Node frontend tests passed**, covering the original app and new chart
  module. Cases include provenance, source/class switching, all seven charts,
  zoom validation/reset, unknown labels, same-name independent files, actual
  controller forwarding, cleared stale state, safe reports and JSON/CSV exports.
- Backend analytics tests cover exact quartiles/histogram counts, sensor-only
  correlations, null/constant/insufficient data, pair counts, invalid cells,
  roundoff/under/overflow, raw-scale inspection without inference, deterministic
  bounded scatter, all-row summaries beyond 1,000 predictions, label alignment,
  ground-truth versus prediction grouping and unavailable reference behavior.
- Reference regeneration was deterministic. The final aggregate JSON is
  **170,455 bytes**, with no exact training vectors. The chart script is about
  26 KB; no chart library, extra installation or frontend build is required.
- Python compilation, JS syntax checks, `pip check` and `git diff --check` passed.
  No additional lint/type-check tool was configured or installed.
- Live HTTP smoke passed: all four class samples, the exact reported Motor case,
  missing/invalid JSON inputs, all nine existing CSV cases and HTTP 200/correct
  MIME types for CSS, app JS, new analytics JS, car WebP and sample CSV.
- Actual browser uploads: the original four-row sample produced one prediction
  per class and its own histogram summaries. Replacing it with a clearly named
  two-row regression fixture cleared the old upload, then showed two Motor
  predictions with the exact supplied values (Voltage 0.80, Vibration 0.92, etc.).
  It did not reuse the old four-row distribution or present original data as upload.
- Browser class filtering returned only the actual Motor sample's values; native
  SVG and keyboard-focus tooltips showed exact counts/proportions. Source change
  to full EDA rendered the original 11,000-row correlations, including −0.50
  Voltage/Temperature, without mixing training-only class analysis into it.
- Seven live boxplot panels rendered. Saved MI/eta/permutation/final-model
  importance values matched the existing analysis, including the 1,540-row
  validation importance scope. No importance was refitted on an upload.
- A range-commit issue was caught in browser testing: filled bounds could remain
  unapplied until a change event. Added **Apply plot ranges**. Verified changing
  X from [0,1] to [0.75,0.85] reduced Motor reference marks from **44 to 22**;
  Reset restored data bounds. Empty/reversed bounds produce a useful error.
- A sticky status strip obscured data during full-panel visual review; removed
  its overlay positioning. The final captured panel has unobstructed chart data.
- Actual HTML report download succeeded for the selected two-row source. The
  inspected report contained its source context, comparison method, two SVG
  charts and **zero script elements**. CSV/JSON contents and escaping are covered
  by actual frontend tests. Browser-native PDF printing itself was not exercised;
  no binary PDF-generation feature is claimed.
- Mobile viewport/document widths both measured **390 px**; chart overflow is
  contained within scrollable panels. Desktop width was 1,366 px with matching
  document width. No browser console errors/warnings were observed. Temporary
  viewport overrides were reset. No quantitative FPS/performance-trace claim is
  made; point counts and mounted chart views are explicitly bounded.

### Files changed in this analytics update

Core: `src/eda/dashboard.py`, `src/api/static/analytics.js`,
`src/api/telemetry.py`, `src/api/datasets.py`, `src/api/schemas.py`,
`src/api/app.py`, and the existing static `index.html`, `app.js`, `styles.css`.
Reference/reproduction: `scripts/build_analytics_reference.py`,
`reports/metrics/experiment_2/dashboard_reference.json`, `.gitignore`.
Verification/documentation: `tests/eda/test_dashboard_analytics.py`,
`tests/frontend/test_analytics.cjs`, `tests/frontend/test_dashboard.cjs`,
`scripts/smoke_test_localhost.py`, `README.md`, and this report.

Other dirty security/reliability files existed before this task and are retained;
their changes are not newly attributed to the chart implementation. Unrelated
Java files and `to/` are untouched. Git HEAD remains `1868ea5`; no Git writes or
deployment occurred. Only `data/raw/.gitkeep` is tracked under raw data. All four
raw CSV fingerprints and saved Random Forest SHA-256 match the prior audit.
No changes appear in historical model metrics, models, training code, report 09
or its figure-generation implementation.

### Run and remaining limits

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8012
```

Open `http://127.0.0.1:8012/#analytics-workspace`. Same Python 3.13 environment
and existing requirements work; no external chart API or service credentials
are needed. Native Windows browser execution was not performed in this audit.
Historical evidence remains distinct from uploaded observations. Scatter is
bounded/quantized as documented, shape warnings are not a validated OOD model,
unknown normalization cannot be inferred, and PDF is browser printing from HTML.
RUL/risk, vehicle-specific validation and service-provider integrations remain
outside this analytics scope. No permission to publish/deploy is implied.
