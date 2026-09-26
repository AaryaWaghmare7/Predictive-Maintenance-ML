# Project Plan

## Current scope

The project uses the EVIoT-PredictiveMaint Dataset from Kaggle. The raw CSV remains in `data/raw/` and is not added to Git. The dataset is documented under the CC BY-NC-SA 4.0 license; sharing or publishing the raw data requires a separate license-compliance check.

## Locked model sequence

1. Model 1: classify `Failure_Probability`, where 0 = No Failure and 1 = Failure.
2. Model 2: classify `Maintenance_Type` as a multiclass target after Model 1 is complete.

## Model 1 current stage

Feature selection is complete. Model 1 uses the fixed 24-column operational allowlist in `src/preprocessing/cleaning.py`. It excludes `Timestamp`, `Failure_Probability`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score`.

The time series retains the chronological earliest-80% / latest-20% split. The first Logistic Regression baseline has been trained and evaluated once on the untouched test partition. No feature engineering, resampling, tuning, threshold changes, or advanced model has been attempted.

## Next approved stage

Review the baseline report and explicitly approve any next experiment. Do not start Model 2 or improve Model 1 without approval.
