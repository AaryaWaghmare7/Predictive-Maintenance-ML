# Target and Feature Review

## Scope and evidence reviewed

This review is read-only. It does not train a model, alter the raw dataset, or change the preprocessing policy.

Evidence reviewed:

- `data/raw/EV_Predictive_Maintenance_Dataset_15min.csv`, loaded with the leading title row skipped.
- `reports/03_Preprocessing_Report.md` and `notebooks/preprocessing/02_Preprocessing.ipynb`.
- `README.md`, `AGENTS.md`, `docs/PROJECT_STRUCTURE.md`, source-code comments, configuration, Git history, and the accompanying `.numbers` file.

The source is the EVIoT-PredictiveMaint Dataset on Kaggle, licensed under CC BY-NC-SA 4.0. Official Kaggle documentation defines `Failure_Probability` values, but the repository does not retain a source URL, data dictionary, sensor-unit reference, maintenance-code mapping, event horizon, or column availability timing. The `.numbers` file is a Numbers archive with internal table files, not a human-readable metadata document. The raw dataset remains outside Git tracking; do not add it to GitHub without separately confirming that the intended sharing complies with the license terms.

## Failure_Probability: observed values and distribution

`Failure_Probability` has `int64` type, no missing values, and exactly two unique values:

| Value | Row count | Proportion |
| ---: | ---: | ---: |
| 0 | 158,061 | 90.118192% |
| 1 | 17,332 | 9.881808% |

The official Kaggle definition is `0 = No Failure` and `1 = Failure`. The column is therefore a binary failure label in this file, despite its probability-like name. It is not a continuous probability, and no calibration claim can be made.

### What the data supports

- The column is the locked Model 1 binary classification target: `0` = No Failure and `1` = Failure.
- It is moderately imbalanced, with 9.88% of records labeled `1`.
- A label value of `1` occurs throughout the series rather than in a single isolated period.

### What the data does not establish

The documentation confirms the class meanings, but does not establish whether a row records a contemporaneous failure, a future failure within a particular horizon, or another time relationship. It also does not identify the affected component, the unit of observation, or how labels were produced. This prevents claims about a future model's prediction horizon or operational deployment readiness.

## Temporal relationship to Timestamp

The timestamps are unique, ordered, and continuous at 15-minute intervals from 2020-01-01 00:00:00 to 2025-01-01 00:00:00. Full-year target rates are stable:

| Year | Rows | Value 1 rows | Value 1 proportion |
| ---: | ---: | ---: | ---: |
| 2020 | 35,136 | 3,468 | 9.8702% |
| 2021 | 35,040 | 3,473 | 9.9115% |
| 2022 | 35,040 | 3,481 | 9.9344% |
| 2023 | 35,040 | 3,418 | 9.7546% |
| 2024 | 35,136 | 3,492 | 9.9385% |

The final timestamp, 2025-01-01 00:00:00, is one incomplete-year row with value `0`; it is not used in the full-year comparison. Across the 60 complete calendar months, the positive rate ranges from 8.9931% to 10.9543%.

Lag correlations of the binary label are close to zero: lag 1 = 0.000530, lag 2 = 0.000145, lag 4 = -0.005298, lag 96 (one day) = -0.002235, and lag 672 (one week) = -0.001468. There are 1,721 adjacent `1`→`1` pairs, close to the 1,712.705 pairs expected from an independent series with the observed positive rate.

This pattern is compatible with approximately independent labels and does not show obvious persistence of a failure state. It does not prove how the data was generated, establish causality, or prove that a chronological split is unnecessary. The existing chronological split remains appropriate because the data is timestamped and availability at prediction time is not documented.

## Excluded columns: values, availability assessment, and evidence

| Column | Observed values/distribution | Evidence and availability assessment | Decision |
| --- | --- | --- | --- |
| `Maintenance_Type` | Integer codes 0–3. Counts: 0 = 122,958, 1 = 26,242, 2 = 17,402, 3 = 8,791. Positive-label rates by code range from 9.6578% to 10.3422%. | No codebook or timestamp-of-recording rule exists. The name suggests a maintenance classification or action, which may be selected after inspection, diagnosis, or the target event. Similar label rates do not prove pre-event availability or safety. | Exclude. |
| `RUL` | Float, 175,393 unique values, range 0.000148–299.999891; lag-1 autocorrelation -0.000364; target correlation 0.000955. | “Remaining useful life” conventionally depends on future degradation/failure timing. The repository does not state when it was calculated. Weak correlation is not evidence that it is available at prediction time. | Exclude. |
| `TTF` | Float, 175,393 unique values, range 0.000009–199.998920; lag-1 autocorrelation -0.000836; target correlation 0.000159. | “Time to failure” is explicitly future/event-relative by name. No source documents a real-time estimator versus a label generated retrospectively. | Exclude. |
| `Component_Health_Score` | Float, 175,393 unique values, range 0.000005–0.999999; lag-1 autocorrelation -0.001480; target correlation -0.001416. | It could be a contemporaneous diagnostic score, but may also be derived from failure outcomes, RUL/TTF, or future lifecycle information. Its calculation and timing are undocumented. | Exclude. |
| `Timestamp` | Unique ordered 15-minute values. | Timestamp establishes temporal ordering and supports a chronological split. It is not used as an initial model feature because raw clock time can proxy collection order and no causal or availability interpretation is documented. | Retain only for auditing and splitting. |

For context, the mean values among labels 0 and 1 are nearly equal for the continuous excluded fields: RUL 216.339508 versus 216.610143, TTF 129.720960 versus 129.750408, and component health 0.744562 versus 0.743305. These numerical comparisons do not overturn the semantic leakage risks.

## Final Model 1 feature set and availability review

Model 1 uses a fixed allowlist of these 24 `float64`, non-missing candidate columns: `SoC`, `SoH`, `Battery_Voltage`, `Battery_Current`, `Battery_Temperature`, `Charge_Cycles`, `Motor_Temperature`, `Motor_Vibration`, `Motor_Torque`, `Motor_RPM`, `Power_Consumption`, `Brake_Pad_Wear`, `Brake_Pressure`, `Reg_Brake_Efficiency`, `Tire_Pressure`, `Tire_Temperature`, `Suspension_Load`, `Ambient_Temperature`, `Ambient_Humidity`, `Load_Weight`, `Driving_Speed`, `Distance_Traveled`, `Idle_Time`, and `Route_Roughness`.

Their names identify common operational or condition variables, but the repository does not prove whether each reading was captured before the label was assigned. The labels below express a conservative availability assessment, not causal claims.

| Assessment | Features | Evidence and remaining limitation |
| --- | --- | --- |
| Clearly operational inputs, subject to confirmation of measurement time | `Battery_Voltage`, `Battery_Current`, `Battery_Temperature`, `Motor_Temperature`, `Motor_Vibration`, `Motor_Torque`, `Motor_RPM`, `Power_Consumption`, `Brake_Pressure`, `Reg_Brake_Efficiency`, `Tire_Pressure`, `Tire_Temperature`, `Suspension_Load`, `Ambient_Temperature`, `Ambient_Humidity`, `Load_Weight`, `Driving_Speed`, `Idle_Time`, `Route_Roughness` | These are named as physical measurements, environmental conditions, load, or driving-state quantities that are ordinarily observable while operating. Each is numeric and complete in this dataset. No data dictionary confirms sensor source, sampling moment, units, or whether any was reconstructed after an event. |
| Potentially leakage-prone cumulative or derived condition inputs | `SoC`, `SoH`, `Charge_Cycles`, `Brake_Pad_Wear`, `Distance_Traveled` | These may be available at the timestamp from onboard counters or estimates, but they can also be lifecycle aggregates or retrospective/derived values. In particular, `SoH` and wear measures can be model-derived; `Charge_Cycles` and distance can be cumulative. The dataset provides no calculation or availability rule. They remain candidate inputs, not confirmed safe features. |
| Uncertain | None beyond the two categories above | Every retained feature has a plausible operational interpretation, but none has source-level timing documentation. This row does not mean the other features are confirmed production-safe. |

Pairwise Pearson correlations with the target are between -0.003277 and 0.003815 for all 24 retained candidates. This makes neither a leakage claim nor a causal claim: correlation alone cannot establish information availability, and weak linear association does not rule out nonlinear or temporal relationships. It does reinforce the need to confirm label provenance before interpreting any later model result.

## Review of the existing preprocessing artifacts

The preprocessing notebook and report correctly:

- Load the titled CSV through the reusable project-relative loader.
- Parse and validate the timestamp while preserving the original schema.
- Treat `Failure_Probability` as the documented binary label: 0 = No Failure and 1 = Failure, rather than a calibrated probability.
- Exclude `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score` conservatively.
- Keep `Timestamp` for chronological ordering and use an earliest-80% / latest-20% split.
- Define, but do not fit, imputation and scaling inside a future scikit-learn pipeline.
- Avoid row deletion, outlier treatment, resampling, feature engineering, and model training.

No new evidence requires a preprocessing change. The current exclusions should remain in place.

## Unresolved uncertainties

1. The class definitions are known, but the observation timing, forecast horizon, and event/component scope of `Failure_Probability=1` remain unknown.
2. No source identifies whether the records represent one vehicle, one component, many vehicles, simulated draws, or independently sampled observations.
3. No column lineage, codebook, sensor units, or feature-availability timing exists.
4. The near-zero temporal autocorrelation across labels and measured columns is unusual for physical telemetry and should be resolved before production-oriented claims.
5. The negative sign of `Battery_Current` may represent discharge, but no sign convention is documented.

## Recommendation for the modeling setup

Do not characterize a future model as an operational failure-risk predictor until the label timing and feature availability are documented. The locked Model 1 setup is:

- Binary classification label: `Failure_Probability`, where 0 = No Failure and 1 = Failure.
- Inputs: the fixed 24-feature allowlist above, with the five cumulative/derived candidates tracked as timing-sensitive.
- Exclusions: `Timestamp` from the initial feature matrix; `Maintenance_Type`, `RUL`, `TTF`, and `Component_Health_Score` as potential leakage.
- Evaluation: chronological 80/20 holdout, with every learned transformation fitted only on the earlier training partition.
- Class handling: report the 9.88% positive prevalence; make no SMOTE decision until a model stage and evaluation goal are agreed.

Before any follow-up experiment, obtain and retain the original dataset page or data dictionary details needed to document the prediction horizon, unit of observation, and per-column availability time.
