# Model 1 Baseline Report

## Scope

This report records the first and only baseline evaluation for Model 1. It compares a majority-class reference model with a balanced Logistic Regression model. No feature engineering, SMOTE, resampling, hyperparameter tuning, threshold tuning, advanced model, or Model 2 work was performed.

## What the model predicts

Model 1 predicts `Failure_Probability`:

- 0 = No Failure
- 1 = Failure

The model uses the fixed 24 approved operational columns as `X`. The target `y` is `Failure_Probability`. `Timestamp`, `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score` are excluded. The raw CSV was read only and remains unchanged.

## Split and data used

Rows were sorted chronologically by `Timestamp` and split without shuffling:

| Partition | Rows | No Failure, 0 | Failure, 1 | Failure share |
| --- | ---: | ---: | ---: | ---: |
| Training: earliest 80% | 140,314 | 126,464 | 13,850 | 9.871% |
| Testing: latest 20% | 35,079 | 31,597 | 3,482 | 9.926% |

Chronological splitting prevents later timestamps from being used to fit earlier predictions. All learned preprocessing is inside the Logistic Regression pipeline and is fit only on the training partition.

## Input features

`SoC`, `SoH`, `Battery_Voltage`, `Battery_Current`, `Battery_Temperature`, `Charge_Cycles`, `Motor_Temperature`, `Motor_Vibration`, `Motor_Torque`, `Motor_RPM`, `Power_Consumption`, `Brake_Pad_Wear`, `Brake_Pressure`, `Reg_Brake_Efficiency`, `Tire_Pressure`, `Tire_Temperature`, `Suspension_Load`, `Ambient_Temperature`, `Ambient_Humidity`, `Load_Weight`, `Driving_Speed`, `Distance_Traveled`, `Idle_Time`, and `Route_Roughness`.

There are 24 input features. The feature list is fixed in `MODEL_1_OPERATIONAL_FEATURES`; no undocumented future column can silently enter the model input.

## Why Logistic Regression is the baseline

Logistic Regression is a suitable first classifier because it is simple, fast, interpretable, and works well with a standard numeric preprocessing pipeline. It provides a clear reference point before considering more complex methods. The model uses `class_weight="balanced"` because the failure class is only about 10% of the data. This weighting changes the loss during training; it is not SMOTE or any other resampling method.

## Model configuration

| Item | Configuration |
| --- | --- |
| Reference model | `DummyClassifier(strategy="most_frequent")` |
| Main baseline | `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)` |
| Preprocessing | Median imputation, then `StandardScaler`, both inside a scikit-learn `Pipeline` |
| Split | Chronological earliest 80% train / latest 20% test |
| Feature count | 24 |
| Resampling | None. SMOTE was not used. |
| Threshold | Default classifier threshold. No threshold tuning. |

## Test-set results

The confusion matrices use rows for actual class `[No Failure, Failure]` and columns for predicted class `[No Failure, Failure]`.

### Majority-class reference

The reference model predicts No Failure for every test row.

| Metric | Value |
| --- | ---: |
| Confusion matrix | `[[31,597, 0], [3,482, 0]]` |
| Accuracy | 0.9007 |
| Precision, class 1 | 0.0000 |
| Recall, class 1 | 0.0000 |
| F1-score, class 1 | 0.0000 |
| ROC-AUC | 0.5000 |
| PR-AUC / Average Precision | 0.0993 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| No Failure, 0 | 0.9007 | 1.0000 | 0.9478 | 31,597 |
| Failure, 1 | 0.0000 | 0.0000 | 0.0000 | 3,482 |

The 90.07% accuracy is misleading because the model detects no failures.

### Balanced Logistic Regression

| Metric | Value |
| --- | ---: |
| Confusion matrix | `[[15,854, 15,743], [1,779, 1,703]]` |
| Accuracy | 0.5005 |
| Precision, class 1 | 0.0976 |
| Recall, class 1 | 0.4891 |
| F1-score, class 1 | 0.1627 |
| ROC-AUC | 0.4912 |
| PR-AUC / Average Precision | 0.0966 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| No Failure, 0 | 0.8991 | 0.5018 | 0.6441 | 31,597 |
| Failure, 1 | 0.0976 | 0.4891 | 0.1627 | 3,482 |

Interpretation of the Logistic Regression confusion matrix:

- It correctly identifies 1,703 of 3,482 failures.
- It misses 1,779 failures.
- It incorrectly labels 15,743 No Failure rows as Failure.
- Its failure precision is close to the test failure prevalence, and its ROC-AUC is below 0.5. On this split, the approved raw features do not show useful linear discriminative signal for this baseline.

## Why class imbalance matters

About nine out of ten rows are No Failure. Accuracy by itself rewards a model that always predicts No Failure, even though that model never finds a failure. Precision, recall, F1-score, ROC-AUC, and Average Precision make the failure-class behavior visible. The balanced Logistic Regression improves recall over the majority reference but creates many false positives and does not improve ranking performance.

## Limitations

- This is one baseline only. No attempt was made to improve it after seeing the test results.
- The data review found near-zero linear relationships and little temporal persistence, which is consistent with the weak baseline result.
- The class meanings are documented, but the prediction horizon, observation timing, component scope, and feature availability timing remain undocumented.
- `SoH`, `Charge_Cycles`, `Brake_Pad_Wear`, and `Distance_Traveled` remain timing-sensitive operational candidates, even though they are in the approved fixed feature set.
- This report does not establish a production-ready predictive-maintenance model.

## Next step

Stop here and review this baseline before authorizing any further work. Any future comparison, feature engineering, threshold policy, class-handling change, or Model 2 work requires explicit approval.
