# Project Decisions

## Experiment 1 data source and license

- Source: EVIoT-PredictiveMaint Dataset on Kaggle.
- License: CC BY-NC-SA 4.0.
- Raw data remains unmodified in `data/raw/` and must not be committed or published to GitHub until the intended use is checked against the license terms.

## Experiment 1 targets

- Model 1 target: `Failure_Probability`.
  - 0 = No Failure.
  - 1 = Failure.
- Model 2 target: `Maintenance_Type` as a multiclass classification task.
  - 0 = None.
  - 1 = Preventive.
  - 2 = Corrective.
  - 3 = Predictive.

## Model 1 feature policy

- Include only the approved 24 operational columns defined by `MODEL_1_OPERATIONAL_FEATURES`.
- Exclude `Timestamp`, `Failure_Probability`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score`.
- Preserve the chronological 80/20 split. Do not shuffle records.
- Do not add feature engineering, scaling outside the future pipeline, resampling, or model training before the corresponding approved stage.

## Model 1 baseline

- The approved baseline is `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)`.
- It uses median imputation and `StandardScaler` inside a scikit-learn pipeline fitted on the chronological training partition only.
- It is compared with `DummyClassifier(strategy="most_frequent")` on the untouched chronological test partition.
- No SMOTE, threshold tuning, hyperparameter tuning, feature engineering, or advanced Model 1 model is approved without a later explicit decision.

## Completed Model 1 experiment and signal review

- Model 1 is documented and paused after finding extremely weak signal in the approved raw features.
- Preserve the existing Random Forest experiment in `notebooks/modeling/02_baseline_model.ipynb`, including its saved outputs. It excludes the same six target/time/outcome fields and uses chronological validation inside the earlier training partition.
- Recorded Random Forest validation results: ROC-AUC 0.502 and Average Precision 0.099. Do not describe these as final test results or rerun training during repository synchronization.
- Preserve the Logistic Regression baseline report and `reports/06_Failure_Signal_Analysis.md` without changing their reported metrics.

## Model 2 feature policy

- Model 2 uses the same fixed 24 operational columns as Model 1.
- Exclude `Timestamp`, `Maintenance_Type`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score`.
- Preserve the chronological 80/20 split. Do not shuffle records.
- Do not train a Model 2 classifier until the Model 2 signal investigation is reviewed and a later decision authorizes a baseline.
- The completed investigation in `reports/07_Maintenance_Type_Signal_Analysis.md` found extremely weak signal; no Model 2 classifier has been trained.
- Experiment 1 is retained as a negative/weak-signal experiment; its existing work is not overwritten by Experiment 2.

## Experiment 2 NEV fault diagnosis

- Experiment 2 is a proof-of-concept EV fault-diagnosis module, supporting Normal / Motor Fault / Inverter Fault / Battery Fault diagnosis.
- User-approved target: `Fault Label`; 0 = Normal, 1 = Motor Fault, 2 = Inverter Fault, 3 = Battery Fault. This is separate from Experiment 1's Maintenance_Type Model 2.
- Preserve the three user-supplied CSVs unchanged under `data/raw/experiment_2/`. Source/license/generation protocol are unconfirmed; do not assume Experiment 1's license applies or publish the NEV raw files.
- Keep exactly seven inputs in original order: Voltage (V), Current (A), Motor Speed (RPM), Temperature (°C), Vibration (g), Ambient Temp (°C), Humidity (%). Exclude Fault Label. No engineering, renaming, row removal or clipping.
- Values are normalized approximately to [0,1], not raw physical units. Accept 1e-12 numerical boundary tolerance without modifying values. Original normalization parameters are unknown, so physical-reading inference is unsupported.
- Preserve supplied 7,700 training / 3,300 final-test membership. Full data supports descriptive EDA/audit only. No new random train/test split; five-fold stratified CV is confined to training.
- Use fixed balanced Logistic Regression, balanced Random Forest (200 trees) and HistGradientBoosting (150 iterations). All learned preprocessing stays inside pipelines fitted on training folds/training only. No hyperparameter/threshold tuning or SMOTE.
- Select by highest mean training-only CV macro F1 before test evaluation; the selected model is Random Forest (CV macro F1 0.994515). Final test accuracy 0.994242, macro F1 0.992082, macro OvR ROC-AUC 0.999698. Do not improve the model against these observed test scores.
- Tree interpretation uses training-only validation permutation importance; importances describe model dependence, not causality. XGBoost/SHAP were unavailable and no new packages were installed; optional SVM was omitted.
- Save the complete inference pipeline and metadata under `models/experiment_2/`. Originally only metadata and approved report figures/statistical snapshots were tracked; the later user-approved collaboration checkpoint intentionally includes the existing small `.joblib` inference pipeline. Raw CSVs and other model binaries remain Git-ignored. The application reuses this saved pipeline without retraining. Reports 08/09/10 document the preserved audit, EDA/signal and comparison.
- Class boundaries are unusually easy to separate: a shallow depth-3 decision tree achieved approximately 0.979 CV macro F1. The extremely high Random Forest performance describes this supplied dataset, not evidence of equivalent real-world OEM performance. This raises realism/provenance concerns, not confirmed synthetic origin or proven target leakage. Unknown upstream full-data normalization and shared groups cannot be ruled out.
- The dataset contains no timestamp or vehicle identifier, therefore temporal generalization and cross-vehicle generalization cannot be claimed.
- Experiment 2 does not provide Remaining Useful Life prediction. RUL will be implemented as a separate module using an appropriate degradation/time-to-failure dataset, with separate approval for that future implementation.
- Experiment 2 modeling was reviewed, approved and published. Its metrics and conclusions remain unchanged in the application phase.

## Localhost application and RUL preparation

- Implement one minimal FastAPI app under the existing `src/api/`, serving a same-origin plain HTML/CSS/JavaScript dashboard. Bind to `127.0.0.1`; no production deployment is approved.
- Load the trusted local NEV pipeline once at startup. Verify its fitted feature order, Random Forest classifier, class mapping and supported package version. No upload may select or replace a model artifact. Missing/invalid artifacts return unavailable status; never automatically retrain.
- Keep exactly the seven original normalized feature aliases for JSON and CSV. Restore canonical feature order, reject extra/missing fields and invalid values, and preserve existing 1e-12 numerical boundary tolerance without clipping or guessing physical-unit scaling.
- Use `model_confidence` for the predicted class probability. It is uncalibrated and is not a future-failure probability, risk score, causal explanation or guarantee of safe operation.
- Validate a CSV in full before inference. Limit it to 5 MiB/20,000 rows and return at most 1,000 row predictions; summary covers every validated row. Uploaded rows have file order only, no inferred timestamps or shared vehicle identity. The app does not retain uploads.
- Display RUL and risk sections as disabled. No RUL/risk endpoint or numerical placeholder is implemented. `src/rul/contracts.py` requires a real future estimate with component, machine ID, documented unit and model version.
- RUL data must have identifiable independent units, ordered degradation observations and a defined end-of-life/failure target. Manually approve provenance/license, censoring and unit/group/time-safe validation before downloading or training. NEV cannot supply this target.
- Future motor, battery and bearing/drivetrain models remain separate adapters with their own feature schema, preprocessing, artifact and units. Do not merge unrelated data row-by-row or claim universal BEV/PHEV/FCEV/manufacturer coverage.
- Collaboration checkpoint approved on 2026-10-08: intentionally track only `models/experiment_2/nev_fault_pipeline.joblib` (2,729,762 bytes; SHA-256 recorded in `INFERENCE_ARTIFACT.md`) so a fresh pull can perform inference without local Mac files or raw datasets. Do not retrain or change recorded metrics. Other artifacts remain ignored.
- Pin the existing model's scikit-learn/pandas/NumPy/joblib versions and tested web dependencies. Use Python 3.13; preserve repository-relative paths and locally selected interpreters. Windows execution still requires confirmation on Tejaswini's machine.
- Current boundary: verify tests and localhost smoke checks, preserve both experiments and unrelated Java files, then commit and safely push to origin/main without force. Stop after synchronization; no RUL training, risk engine, cloud deployment or major UI redesign is authorized.

## Collaborator environments

- Shared VS Code settings must not specify a platform-specific interpreter path or force Conda.
- Each collaborator selects their own Python interpreter and notebook kernel locally.
- Each collaborator separately obtains the same raw CSV at `data/raw/EV_Predictive_Maintenance_Dataset_15min.csv`; Git tracks only `data/raw/.gitkeep` in that directory.

## Open items

- Confirm the label's precise observation timing and any prediction horizon.
- Confirm measurement timing and derivation for every feature, especially cumulative or condition-score fields.
- For NEV, obtain its source/license, original normalization parameters and label-generation/measurement protocol.
- Obtain an independent timestamp/vehicle-aware evaluation dataset before real-world diagnosis claims; use a separate degradation dataset for future RUL work.
