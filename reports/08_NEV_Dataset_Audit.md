# Experiment 2: NEV Dataset Audit

## Scope and provenance

This audit covers the three user-supplied NEV CSV files, copied byte-for-byte
into `data/raw/experiment_2/`. The supplied originals and Experiment 1 data,
reports, tests and conclusions were not modified. No rows were dropped,
deduplicated, clipped or resampled. Audit results are also saved in
`reports/metrics/experiment_2/audit.json`.

The user documents `Fault Label` as 0 = Normal, 1 = Motor Fault,
2 = Inverter Fault, 3 = Battery Fault. No original dataset source page,
generation protocol, normalization parameters or license was supplied with
these CSVs. Do not assume Experiment 1's Kaggle license applies to NEV or
publish these raw files without checking their own license.

## Approved interpretation and scope

Experiment 2 is a **proof-of-concept EV fault-diagnosis module** supporting
Normal / Motor Fault / Inverter Fault / Battery Fault diagnosis. Class
boundaries are unusually easy to separate: a shallow depth-3 decision tree
achieved approximately **0.979 CV macro F1** on training-only folds.
Random Forest's final supplied-test results are **Accuracy = 0.994242,
Macro F1 = 0.992082, OvR ROC-AUC = 0.999698**. These describe performance on
this supplied dataset, **not evidence of equivalent real-world OEM
performance**. No timestamp or vehicle identifier is available, so temporal
generalization and cross-vehicle generalization cannot be claimed.

The inputs are already normalized approximately to [0,1]; original
physical-unit normalization parameters are not currently known. Experiment 2
**does not provide Remaining Useful Life prediction**. RUL will be implemented
as a separate module using an appropriate degradation/time-to-failure dataset.
The audit and approved reports are published without changing recorded
metrics, retraining models or publishing raw CSVs.

## Shapes, schema and classes

| File | Rows | Columns | Normal (0) | Motor (1) | Inverter (2) | Battery (3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEV_fault_dataset.csv | 11,000 | 8 | 5,000 | 2,000 | 2,000 | 2,000 |
| NEV_fault_training_dataset.csv | 7,700 | 8 | 3,500 | 1,400 | 1,400 | 1,400 |
| NEV_fault_testing_dataset.csv | 3,300 | 8 | 1,500 | 600 | 600 | 600 |

All three partitions have identical proportions: Normal 45.4545%; each fault
class 18.1818%. The supplied split is 70% training / 30% testing. It is not
the chronological 80/20 split from Experiment 1, and no new split was made.

Exact columns, with original spelling and ordering:

1. Voltage (V)
2. Current (A)
3. Motor Speed (RPM)
4. Temperature (°C)
5. Vibration (g)
6. Ambient Temp (°C)
7. Humidity (%)
8. Fault Label

All eight columns load as `float64`. Target values are exactly
`0.0, 1.0, 2.0, 3.0`; validated label copies are cast to integer in memory.
Unlike Experiment 1, these CSVs start with the actual header, not a title row.
The existing loader handles both formats. There are no timestamp or vehicle-ID
columns. Original column names are retained without implying physical units.

## Data quality and compatibility

| Check | Full | Training | Testing |
| --- | ---: | ---: | ---: |
| Missing values, all columns | 0 | 0 | 0 |
| Infinite values | 0 | 0 | 0 |
| Exact duplicate rows | 0 | 0 | 0 |
| Duplicate seven-feature observations | 0 | 0 | 0 |
| Values outside [0,1] beyond 1e-12 tolerance | 0 | 0 | 0 |

Training and testing have compatible schemas, numeric types and all four
classes. Their exact eight-column row overlap is **0**, and their exact
seven-feature overlap is **0**. A multiset comparison, including row
multiplicity and labels, confirms that training plus testing exactly
reconstructs the full dataset. Concatenation order does not need to match.

This does not establish independence between vehicles, trips, experiments or
near-duplicate measurements. Without identifiers or provenance, related
observations could still be spread across the supplied files.

## Normalized values, not physical measurements

All seven features lie approximately in **[0,1]**. Values must not be presented
as raw volts, amps, RPM, degrees Celsius, acceleration in g or humidity
percentages. The original conversion formula and raw physical measurements
are unavailable, so physical thresholds cannot be recovered.

Two full-dataset values are `1.0000000000000002`: one Current value in training
and one Vibration value in testing. Their 2.22e-16 excess is consistent with
floating-point roundoff. Validation accepts a documented 1e-12 boundary
tolerance and leaves these values unchanged. There are no larger violations.

Full-dataset descriptive statistics below are normalized numbers. Sample SD
uses pandas' `ddof=1`. Min/max round to 0/1 for every full-dataset feature.

| Feature | Min | Max | Mean | SD | 5% | 25% | Median | 75% | 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Voltage (V) | 0 | 1 | 0.728576 | 0.224415 | 0.202010 | 0.612900 | 0.797613 | 0.891521 | 0.971709 |
| Current (A) | 0 | 1 | 0.431700 | 0.245215 | 0.092616 | 0.237031 | 0.396222 | 0.604094 | 0.897279 |
| Motor Speed (RPM) | 0 | 1 | 0.443010 | 0.256147 | 0.083001 | 0.229121 | 0.413228 | 0.640606 | 0.898823 |
| Temperature (°C) | 0 | 1 | 0.380225 | 0.229912 | 0.044850 | 0.197145 | 0.370151 | 0.520455 | 0.844871 |
| Vibration (g) | 0 | 1 | 0.281481 | 0.240546 | 0.022556 | 0.109902 | 0.217818 | 0.345722 | 0.847435 |
| Ambient Temp (°C) | 0 | 1 | 0.495347 | 0.286040 | 0.051457 | 0.249639 | 0.490943 | 0.743677 | 0.943544 |
| Humidity (%) | 0 | 1 | 0.500356 | 0.290172 | 0.050757 | 0.246695 | 0.503985 | 0.749887 | 0.950953 |

Exact per-file ranges, means, SD and quantiles are retained in `audit.json`.
Some test values reach endpoints absent from training (for example Ambient
Temp 0 and 1). They are retained; no train-range clipping is applied.

## Outlier findings and decision

IQR flags use training-only fences `Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`, then count
observations in each supplied file. These are diagnostic flags, not evidence
of faulty measurements.

| Feature | Training lower / upper fence | Full flags | Train flags | Test flags |
| --- | --- | ---: | ---: | ---: |
| Voltage (V) | 0.202191 / 1.302537 | 551 | 388 | 163 |
| Vibration (g) | -0.246802 / 0.703138 | 1,064 | 740 | 324 |
| All other five features, individually | See audit.json | 0 | 0 | 0 |

Low Voltage and high Vibration occur in class-dependent ranges. Removing or
winsorizing them could erase useful diagnostic information. **No outlier
treatment is applied**, and no row is dropped or silently repaired.

## Leakage and split limitations

`Fault Label` is strictly excluded from the seven-column input allowlist.
There are no explicit future-outcome, RUL or maintenance-result fields in this
schema. Measurement and label timing are still undocumented, so sensor
availability before a fault cannot be assumed. The intended task is diagnosis
of a fault state, not forecasting a future failure.

The exact preservation of proportions is consistent with a stratified-like
split but does not prove its generating method. Upstream normalization may
have used the full dataset: all full-dataset feature ranges reach 0 and 1.
That is a possibility, not established leakage. Our code cannot undo an
unknown upstream transformation. It fits its own imputation/scaling only on
training folds or the supplied training file.

No temporal, cross-vehicle or real-world generalization claim is justified.

## Raw-data fingerprints

SHA-256 was checked after the audit/training run and matches the supplied
originals and local copies:

| File | SHA-256 |
| --- | --- |
| Full | 58892b061bbd314df4161ce52dffa51f8a2bdabe1074fb12a867ce7fdbce3ac0 |
| Training | 858d2bd5ad59754a7ad4e9b510bb4adce67627b8a672e828429e8e7b11017d64 |
| Testing | 49c0f7f940e0cc6018fed5fa6c76cbf747ca3e9e50702ac1400a8ec1c755ab2c |

## Reproduction

From the repository root: `python scripts/run_nev_experiment.py`. This is the
approved full experiment, including training and final evaluation; do not
repeatedly rerun it to optimize observed test scores. Files default to
`data/raw/experiment_2/`; `--data-dir` accepts another directory containing all
three original filenames. No absolute personal paths are embedded in code.
