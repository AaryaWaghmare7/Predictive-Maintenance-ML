# Project Decisions

## Data source and license

- Source: EVIoT-PredictiveMaint Dataset on Kaggle.
- License: CC BY-NC-SA 4.0.
- Raw data remains unmodified in `data/raw/` and must not be committed or published to GitHub until the intended use is checked against the license terms.

## Targets

- Model 1 target: `Failure_Probability`.
  - 0 = No Failure.
  - 1 = Failure.
- Model 2 target: `Maintenance_Type` as a multiclass classification task.

## Model 1 feature policy

- Include only the approved 24 operational columns defined by `MODEL_1_OPERATIONAL_FEATURES`.
- Exclude `Timestamp`, `Failure_Probability`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score`.
- Preserve the chronological 80/20 split. Do not shuffle records.
- Do not add feature engineering, scaling outside the future pipeline, resampling, or model training before the corresponding approved stage.

## Model 1 baseline

- The approved baseline is `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)`.
- It uses median imputation and `StandardScaler` inside a scikit-learn pipeline fitted on the chronological training partition only.
- It is compared with `DummyClassifier(strategy="most_frequent")` on the untouched chronological test partition.
- No SMOTE, threshold tuning, hyperparameter tuning, feature engineering, advanced model, or Model 2 work is approved without a later explicit decision.

## Open items

- Confirm the label's precise observation timing and any prediction horizon.
- Confirm measurement timing and derivation for every feature, especially cumulative or condition-score fields.
