# Preprocessing Report

## Scope

This report documents preparation for a future failure-state classification task. No model was trained, no resampling method was used, and the raw dataset was not changed.

## Dataset loaded

- Source: `data/raw/EV_Predictive_Maintenance_Dataset_15min.csv`
- Shape after loading: 175,393 rows and 30 columns
- Time coverage: 2020-01-01 00:00:00 through 2025-01-01 00:00:00
- Sampling cadence: continuous 15-minute intervals
- Loader finding: the first line is a dataset title rather than the header. The reusable loader detects this form and reads the CSV with `skiprows=1`.

## Target decision

`Failure_Probability` is the locked Model 1 target. Official Kaggle documentation for the EVIoT-PredictiveMaint Dataset defines `0` as No Failure and `1` as Failure. The observed values are binary integers:

| Value | Rows | Share |
| --- | ---: | ---: |
| 0 | 158,061 | 90.12% |
| 1 | 17,332 | 9.88% |

Despite its name, this column is not a continuous probability in this file. It is a binary failure label, not a calibrated probability. The class imbalance is documented but no SMOTE, class weighting, undersampling, or other balancing was applied in preprocessing.

## Missing values, duplicates, and types

- Missing values: 0 in every column.
- Duplicate complete rows: 0.
- Duplicate timestamps: 0.
- Invalid timestamps after parsing: 0.
- Numeric data: 29 numeric columns. `Failure_Probability` and `Maintenance_Type` are integer-coded.
- `Timestamp`: parsed to `datetime64` in memory while retaining its original column name.

No rows or columns were dropped for missingness or duplication because the checks found no such problems.

## Feature selection and leakage prevention

The final Model 1 input set has 24 approved sensor and operating columns. A fixed allowlist is used so undocumented future columns cannot silently enter `X`. The following fields are excluded before future model fitting:

| Excluded field | Decision |
| --- | --- |
| `Failure_Probability` | Target label. |
| `Timestamp` | Used for ordering, validation, and splitting. It is not an initial model feature. |
| `Maintenance_Type` | Potential maintenance action/result recorded at or after the outcome. Also lacks a codebook. |
| `RUL` | Potential future/prognostic outcome. |
| `TTF` | Potential future/prognostic outcome. |
| `Component_Health_Score` | Potential derived/post-outcome health measure. |

This is a conservative exclusion policy. It should be revisited only after the data source documents when each value becomes available relative to the failure label. The EDA also found near-zero pairwise correlations between the binary label and all available numeric fields, which raises a data-generating-process concern but does not itself prove leakage or an error.

## Timestamp and split strategy

The dataset is a single ordered 15-minute time series. A random shuffle would allow future observations into training, so future modeling should use a chronological holdout:

- Training: earliest 80% of rows.
- Test: latest 20% of rows.
- Sorting: stable ascending sort by `Timestamp` immediately before the split.
- Transformation fitting: fit imputation and scaling on training data only, then transform the test data.

The reusable `chronological_train_test_split` helper implements this policy. It does not create a model split artifact during preprocessing.

## Outliers

IQR screening flags approximately 13% to 15% of observations in most numeric columns. The values remain within the dataset's declared-looking bounded ranges and there are no missing or malformed values to repair. These flags likely reflect broad simulated distributions rather than proven measurement errors.

Decision: no clipping, winsorization, or row removal. Any robust scaling or domain-specific thresholds must be justified during future modeling after target semantics and sensor units are confirmed.

## Transformations prepared

- Load CSV while safely handling a leading title row.
- Parse and validate `Timestamp` without renaming it.
- Select the 24 non-leakage candidate features without fitting or altering values.
- Define an unfitted scikit-learn `ColumnTransformer`: median imputation followed by standard scaling for numeric features.

The transformer remains unfitted. A later training workflow must combine it with an estimator in a scikit-learn `Pipeline` and fit the full pipeline only on the chronological training partition.

## Deliberately not done

- No ML model training or evaluation.
- No raw-data modification or deletion.
- No feature engineering, including time-derived or rolling-window features.
- No target recoding beyond treating observed 0/1 values as a binary label.
- No missing-value dropping, duplicate removal, outlier removal, SMOTE, or other resampling.
- No fitting of imputation, scaling, or any learned transformation on all data.

## Remaining uncertainties

1. The label meanings are confirmed, but the Kaggle documentation has not yet established the exact observation timing, prediction horizon, or affected component for each row.
2. A codebook and availability timing are needed for `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score` before any could be considered safe Model 1 features.
3. The single series has almost no short-term autocorrelation in the measured columns, which is unusual for physical 15-minute sensor telemetry. Confirm whether records were synthetically generated or independently sampled before interpreting model results as real-world predictive maintenance performance.

## Next step

The approved chronological Logistic Regression baseline has now been completed. See `reports/05_Baseline_Model_Report.md`. Do not begin another experiment, feature-engineering step, or Model 2 without explicit approval.
