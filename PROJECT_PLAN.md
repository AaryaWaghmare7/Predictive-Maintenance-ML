# Project Plan

## Current scope

The project uses the EVIoT-PredictiveMaint Dataset from Kaggle. The raw CSV remains in `data/raw/` and is not added to Git. The dataset is documented under the CC BY-NC-SA 4.0 license; sharing or publishing the raw data requires a separate license-compliance check.

## Locked model sequence

1. Model 1: classify `Failure_Probability`, where 0 = No Failure and 1 = Failure.
2. Model 2: classify `Maintenance_Type` as a multiclass target after Model 1 is complete.

## Model 1 current stage

Model 1 is documented and paused. It uses the fixed 24-column operational allowlist in `src/preprocessing/cleaning.py`. It excludes `Timestamp`, `Failure_Probability`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score`.

The time series retains the chronological earliest-80% / latest-20% split. The first Logistic Regression baseline and a failure-signal investigation are documented. No feature engineering, resampling, tuning, threshold changes, or advanced model has been attempted.

## Model 2 current stage

The Model 2 target is `Maintenance_Type`: 0 = None, 1 = Preventive, 2 = Corrective, and 3 = Predictive. Its signal investigation uses the same fixed 24-column operational allowlist. It excludes `Timestamp`, `Maintenance_Type`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score`.

No Model 2 classifier has been trained. The signal investigation must be reviewed before any multiclass baseline is authorized.

## Next approved stage

Review the Model 2 signal report and explicitly approve any next experiment. Do not improve Model 1 or train Model 2 without approval.
