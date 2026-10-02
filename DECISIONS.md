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
- Dataset 2 has not started.

## Collaborator environments

- Shared VS Code settings must not specify a platform-specific interpreter path or force Conda.
- Each collaborator selects their own Python interpreter and notebook kernel locally.
- Each collaborator separately obtains the same raw CSV at `data/raw/EV_Predictive_Maintenance_Dataset_15min.csv`; Git tracks only `data/raw/.gitkeep` in that directory.

## Open items

- Confirm the label's precise observation timing and any prediction horizon.
- Confirm measurement timing and derivation for every feature, especially cumulative or condition-score fields.
