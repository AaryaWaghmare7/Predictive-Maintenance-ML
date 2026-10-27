# Project Plan

## Current scope

Experiment 1 uses the EVIoT-PredictiveMaint Dataset from Kaggle and is retained as a negative/weak-signal experiment. The raw CSV remains in `data/raw/` and is not added to Git. That dataset is documented under the CC BY-NC-SA 4.0 license; sharing or publishing the raw data requires a separate license-compliance check.

Experiment 2 is a proof-of-concept EV fault-diagnosis module using the user-supplied NEV full/training/testing CSVs. Its own source, license and normalization protocol remain unconfirmed. Original files are preserved locally in `data/raw/experiment_2/` and are not published.

## Experiment 1 locked model sequence

1. Model 1: classify `Failure_Probability`, where 0 = No Failure and 1 = Failure.
2. Model 2: classify `Maintenance_Type` as a multiclass target after Model 1 is complete.

## Model 1 current stage

Model 1 is documented and paused. It uses the fixed 24-column operational allowlist in `src/preprocessing/cleaning.py`. It excludes `Timestamp`, `Failure_Probability`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score`.

The time series retains the chronological earliest-80% / latest-20% split. The first Logistic Regression baseline and a failure-signal investigation are documented. Tejaswini's existing `notebooks/modeling/02_baseline_model.ipynb` also records a Random Forest experiment with a chronological validation split inside the training partition. Its saved validation ROC-AUC is 0.502 and Average Precision is 0.099; these are not final test metrics. Both investigations show extremely weak signal in the approved raw features. Experiment 2 does not retrain, improve or overwrite this work.

## Model 2 current stage

The Model 2 target is `Maintenance_Type`: 0 = None, 1 = Preventive, 2 = Corrective, and 3 = Predictive. Its signal investigation uses the same fixed 24-column operational allowlist. It excludes `Timestamp`, `Maintenance_Type`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score`.

The signal investigation is complete and documented in `reports/07_Maintenance_Type_Signal_Analysis.md`. It found extremely weak signal. No Model 2 classifier has been trained, and any multiclass baseline requires explicit approval.

## Experiment 2 completed stage

Target: `Fault Label`, supporting Normal / Motor Fault / Inverter Fault / Battery Fault diagnosis, with codes 0 / 1 / 2 / 3. Inputs are the seven unchanged sensor columns, defined in `src/data/nev_fault.py`, already normalized approximately to [0,1]. Original physical-unit normalization parameters are not currently known. The full file is for descriptive EDA and validation; development uses only the supplied training file and final evaluation uses only the supplied testing file.

The audit confirms 11,000 rows, with a supplied 7,700/3,300 split, no missing or exact duplicate rows, no exact train/test overlap, and exact reconstruction of the full dataset. Reports 08 and 09 document normalization, class associations, sharp rule-like boundaries and unresolved provenance. No new random train/test split, engineered features, resampling or test-driven tuning was used.

The approved fixed Logistic Regression, Random Forest and HistGradientBoosting comparison is complete. Training-only five-fold macro F1 selected Random Forest before test evaluation. Its final supplied-test accuracy is 0.994242, macro F1 0.992082 and macro OvR ROC-AUC 0.999698. Report 10 records all candidate scores, per-class metrics, confusion matrices and training-validation feature importance. The complete selected inference pipeline and metadata are saved locally under `models/experiment_2/`.

Class boundaries are unusually easy to separate: a shallow depth-3 decision tree achieved approximately 0.979 CV macro F1. The extremely high Random Forest results describe this supplied dataset, not evidence of equivalent real-world OEM performance. Model metadata and report figures/statistical snapshots are published. The user-approved localhost checkpoint now intentionally includes the existing 2,729,762-byte inference pipeline; raw CSVs and other model binaries remain Git-ignored. This storage-policy exception does not change the experiment or its metrics.

## Localhost application and RUL preparation

The approved application phase reuses the existing saved NEV Random Forest without retraining or tuning. A FastAPI app in `src/api/` serves a plain HTML/CSS/JavaScript dashboard, `/health`, `/predict-fault` and `/analyze-csv`. Inputs remain exactly the seven normalized sensors; parameter learning and physical-unit conversion are not performed at request time. Missing or incompatible artifacts make prediction unavailable rather than triggering training.

CSV rows are validated and treated as independent observations, not a temporal sequence. Class confidence is an uncalibrated model output, not a real-world failure probability or risk rating. RUL and risk controls are visible but disabled, with no numerical outputs.

The 2026-10-09 dashboard/CSV improvement keeps this architecture. A responsive
engineering workspace separates sensor input, condition/confidence and system
information. Four supplied samples fill inputs without automatically predicting.
The CSV endpoint requires the seven sensors, accepts UTF-8/BOM and trimmed
header whitespace, and reports/ignores extra metadata rather than feeding it to
the model. All sensor rows are validated before inference; errors name missing
columns or invalid line/feature. No normalization parameters are fitted at
inference and uploaded files are not rewritten.

`src/rul/` defines future prediction contracts only. `data/rul/`, `models/rul/` and `reports/rul/` reserve separate locations. Dataset requirements are documented in `reports/11_RUL_Dataset_Requirements.md`; component routing and coverage limits are documented in `docs/MODULAR_ARCHITECTURE.md`. No RUL dataset has been downloaded or approved and no RUL model is trained. Motor, battery and bearing/drivetrain models will have their own schemas, preprocessing, units and validation rather than merging unrelated datasets.

## Review boundary and future work

2026-10-27: the user approved publishing the tested **EVision collaboration
checkpoint** to `origin/main`, including the existing inference artifact and
local frontend assets, with Windows PowerShell instructions. Verify current
tests/browser flows, preserve and integrate the README-only remote commit, and
stage only relevant app/documentation/test changes. No force push, deployment,
model retraining, raw-data publication or RUL/risk implementation is authorized.
This supersedes earlier local-only Git boundaries, not the research/licensing
limitations. Native Windows testing remains Tejaswini's next validation step.

2026-10-26 showcase audit: the local saved-model demo, interactive CSV analytics,
request-size guard, readiness check and professional documentation are prepared
for review. No new training is authorized. Confirm NEV provenance/license and
artifact redistribution permissions, approve publication, and reconcile the
README-only remote change before a public GitHub/LinkedIn release. Deployment
remains unapproved; prefer the documented single-origin FastAPI/static service.

2026-10-15 clarification: complete the existing black/charcoal dashboard and
fault/CSV checks only. The collaborator's RUL dataset is not in this checkout;
do not download, extract or train on a replacement. RUL and Maintenance Risk
stay visibly pending without estimates, metrics, thresholds or risk levels.
The existing fault classifier and evaluation helpers remain unchanged.

Experiment 2's completed modeling work and the localhost MVP/RUL-scaffolding
checkpoint (dc2ee42) are already published, including the existing small
inference artifact. The dashboard/CSV repair was initially a **local review only**;
the 2026-10-27 approval above now permits its tested collaboration publication.
Run the complete suite, frontend checks and real application uploads before
publishing. Use Python 3.13 and pinned dependencies; keep
Windows startup notes and the pytest temp-directory workaround. Experiment 1
remains paused; reports, metrics, artifacts and raw data are preserved. Obtain
NEV source/license, normalization parameters, labeling protocol and independent
vehicle/time-aware validation before real-world claims.

Experiment 2 does not provide Remaining Useful Life prediction. RUL requires a manually approved degradation/time-to-failure dataset and separate training approval. The NEV dataset has no timestamp or vehicle identifier, therefore temporal generalization and cross-vehicle generalization cannot be claimed. BEV/PHEV/FCEV or multi-manufacturer support also requires appropriate training and validation evidence; it is not established by the current dataset.
