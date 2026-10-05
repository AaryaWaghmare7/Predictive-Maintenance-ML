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

Class boundaries are unusually easy to separate: a shallow depth-3 decision tree achieved approximately 0.979 CV macro F1. The extremely high Random Forest results describe this supplied dataset, not evidence of equivalent real-world OEM performance. Model metadata and report figures/statistical snapshots are published; the trained pipeline and raw CSVs remain Git-ignored.

## Review boundary and future work

Experiment 2 has been reviewed and approved for repository publication. Safely commit and synchronize the approved work without retraining or changing metrics. Stop after synchronization; new experiments still require approval. Experiment 1 remains paused. No website/API is implemented. Obtain NEV source/license, normalization parameters, labeling protocol and independent vehicle/time-aware validation before real-world claims.

Experiment 2 does not provide Remaining Useful Life prediction. RUL will be implemented as a separate module using an appropriate degradation/time-to-failure dataset; that future implementation requires separate approval. The NEV dataset has no timestamp or vehicle identifier, therefore temporal generalization and cross-vehicle generalization cannot be claimed.
