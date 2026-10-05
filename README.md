# PREDICTIVE MAINTENANCE - ML

A beginner-friendly machine-learning project that uses EV sensor data to
support predictive-maintenance analysis. The project focuses on a repeatable
workflow: inspect data, clean it, explore it, choose a target, and only then
train and evaluate a model.

> Tesla may be used as an industry case study only. This project uses no Tesla
> proprietary data.

## Project status

| Stage | Status |
| --- | --- |
| Experiment 1: inspection, preprocessing and EDA | Complete; retained as a negative/weak-signal experiment |
| Experiment 1, Model 1: Failure_Probability | Logistic Regression baseline, Random Forest notebook experiment, and signal investigation complete; paused |
| Experiment 1, Model 2: Maintenance_Type | Signal investigation complete; no classifier trained |
| Experiment 2: NEV Fault Label | Proof-of-concept fault-diagnosis module complete; reviewed and approved for the repository |

The Experiment 1 dataset contains 175,393 records and 30 columns. The first
preprocessing run found no missing values and no exact duplicate rows.
Experiment 2 uses 11,000 observations and seven normalized sensor features
for four-class fault **diagnosis**, not Remaining Useful Life (RUL) prediction.

## Setup

Use your own Python environment. The shared VS Code settings do not force an
interpreter path, Conda, or a package manager. In VS Code, run **Python: Select
Interpreter** from the Command Palette and choose your local environment.
For notebooks, also choose that environment in the kernel selector. Do not
commit a personal absolute interpreter path to workspace settings.

If creating a new environment, use these commands from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Keep original datasets in `data/raw/`. Cleaned datasets are written to
`data/processed/`. Both folders are ignored by Git, so large data files are
kept local and never pushed to GitHub.

Every collaborator must separately download or copy the same
EVIoT-PredictiveMaint dataset from Kaggle and place the original file at:

```text
data/raw/EV_Predictive_Maintenance_Dataset_15min.csv
```

GitHub stores only `data/raw/.gitkeep`, not the CSV. Pulling the repository
does not download the dataset. Keep its title row and actual header unchanged;
the project loader handles the title row. The source is documented under
CC BY-NC-SA 4.0; do not publish the raw data without checking the intended use
against the license terms.

To inspect the first dataset in `data/raw/`:

```powershell
python scripts/inspect_dataset.py
```

## Experiment 1 preprocessing

The preprocessing script performs safe, repeatable cleaning:

- standardizes column names to `snake_case`
- converts and sorts the `timestamp` column
- replaces infinite numeric values with missing values
- removes exact duplicate rows
- stops if missing values remain, so an imputation choice is made explicitly
- writes a JSON data-quality report

Run it with a dataset already placed in `data/raw/`:

```powershell
python scripts/preprocess_dataset.py
```

Or provide an explicit file path:

```powershell
python scripts/preprocess_dataset.py --input "C:\path\to\EV_Predictive_Maintenance_Dataset_CSV.csv"
```

Outputs:

- `data/processed/ev_predictive_maintenance_cleaned.csv`
- `reports/metrics/preprocessing_report.json`

## Experiment 1: Model 1 baseline

Model 1 predicts `Failure_Probability`, where 0 means No Failure and 1 means Failure. It uses the fixed 24-column operational feature set and a chronological 80/20 split. The first baseline compares a majority-class reference with balanced Logistic Regression. See `reports/05_Baseline_Model_Report.md` for the full test-set evaluation.

The recorded Logistic Regression results are ROC-AUC 0.4912 and Average
Precision 0.0966. The failure-signal investigation in
`reports/06_Failure_Signal_Analysis.md` found extremely weak signal in the
approved raw features, and Model 1 is paused.

Tejaswini's existing `notebooks/modeling/02_baseline_model.ipynb` also contains
a Random Forest experiment. Its saved validation results are ROC-AUC 0.502
and Average Precision 0.099. It uses an earlier/later validation split within
the training partition; these are validation results, not final test results.
The notebook and its saved outputs are retained as completed experimental
work. Its dataset path is resolved relative to the repository on both macOS
and Windows. It expects the local processed CSV listed in the preprocessing
section, which must be generated separately from the original raw CSV.
These results are retained without retraining during Experiment 2.

For reference, the existing baseline entry point is:

```powershell
python scripts/train_baseline.py
```

## Experiment 1: Model 2 status

Model 2 uses `Maintenance_Type`: 0 = None, 1 = Preventive, 2 = Corrective, and 3 = Predictive. Its signal investigation uses the same fixed 24 operational features as Model 1 and excludes `Timestamp`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score` to avoid leakage. See `reports/07_Maintenance_Type_Signal_Analysis.md` before authorizing a multiclass baseline.

The investigation found extremely weak signal. No Model 2 classifier has been
trained. Experiment 1 remains paused and its original conclusions are preserved.

The completed Model 1 and Model 2 signal plots in `reports/figures/` are tracked
alongside their reports. Experiment 1 generated data, metrics, and model
artifacts remain local and ignored by Git.

## Experiment 2: NEV fault diagnosis

Target: `Fault Label`. Mapping supplied by the user:
**0 = Normal, 1 = Motor Fault, 2 = Inverter Fault, 3 = Battery Fault**.
This is a separate experiment, not Experiment 1's Maintenance_Type Model 2.
Experiment 2 is a **proof-of-concept EV fault-diagnosis module**.

Each collaborator must separately supply the same three original files:

```text
data/raw/experiment_2/NEV_fault_dataset.csv
data/raw/experiment_2/NEV_fault_training_dataset.csv
data/raw/experiment_2/NEV_fault_testing_dataset.csv
```

The full file is for descriptive EDA/audit. The supplied **7,700 training /
3,300 testing** membership is preserved; there is no new random train/test
split. Five-fold stratified validation is confined to training. Raw NEV files
are ignored by Git. Their own source/license and normalization protocol still
need confirmation; Experiment 1's license must not be assumed to apply.

Inputs, in the inference order, are `Voltage (V)`, `Current (A)`,
`Motor Speed (RPM)`, `Temperature (°C)`, `Vibration (g)`,
`Ambient Temp (°C)`, `Humidity (%)`. Despite the physical units in these names,
the numbers are already normalized to approximately [0,1]. Original
physical-unit normalization parameters are not currently known, so no
physical-unit conversion is available. `Fault Label` never enters X.

| Fixed candidate | Final test accuracy | Final test macro F1 | Macro OvR ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.980909 | 0.975324 | 0.999191 |
| Random Forest | 0.994242 | 0.992082 | 0.999698 |
| HistGradientBoosting | 0.993333 | 0.990832 | 0.999759 |

**Random Forest was selected before test evaluation**, using training-only
CV macro F1. The full inference pipeline is local at
`models/experiment_2/nev_fault_pipeline.joblib`, with `metadata.json` beside
it. Metadata records feature order, class mapping, versions, fingerprints,
configurations, CV scores and final test results. Metadata is tracked;
the trained `.joblib` pipeline remains local and Git-ignored. Pulling code
does not download a fitted model. Collaborators who need inference must
separately obtain the trusted pipeline and use compatible package versions.

Reports:

- [Dataset audit](reports/08_NEV_Dataset_Audit.md)
- [EDA and leakage/signal review](reports/09_NEV_EDA_Report.md)
- [Model comparison, confusion matrices and interpretation](reports/10_NEV_Model_Comparison.md)

The approved full-run command is:

```powershell
python scripts/run_nev_experiment.py
python -m pytest -q
```

Do not repeatedly rerun/tune against the observed test results. No SMOTE,
feature engineering or hyperparameter/threshold search was performed.
The eight report figures and four statistical/metric snapshots under
`reports/figures/experiment_2/` and `reports/metrics/experiment_2/` are tracked
with the approved reports. They can be reproduced through the script, which
also retrains models; do not run it merely to receive the latest code.

Local inference with the saved pipeline (from the repository root):

```python
import joblib
from src.prediction.nev_fault import predict_nev_faults

pipeline = joblib.load("models/experiment_2/nev_fault_pipeline.joblib")
# sensors: DataFrame with exactly the seven normalized input columns above.
predictions = predict_nev_faults(pipeline, sensors)
```

Very sharp class-specific ranges and a depth-3 diagnostic tree's CV macro F1
of approximately 0.979 suggest unusually easy, rule-like class boundaries.
The high Random Forest scores describe performance on this supplied dataset,
**not evidence of equivalent real-world OEM performance**.
Realism/label-generation provenance is
unconfirmed. No timestamps or vehicle IDs are present, so the scores establish
neither temporal nor cross-vehicle generalization. Original normalization may
also have used full-dataset information; our own pipeline fitting is training
only. The model does not accept raw volts/amps/RPM by guessing a conversion.

Experiment 2 **does not provide Remaining Useful Life (RUL) prediction**.
RUL will be implemented as a separate module using an appropriate
degradation/time-to-failure dataset. No website/API has been implemented.
Experiment 2 has been reviewed and approved for repository publication;
this approval does not authorize new modeling experiments.

## Collaboration workflow

To receive the latest code in an existing clone, first check for local changes:

```powershell
git status
git switch main
git pull --ff-only origin main
python -m pytest -q
```

Commit legitimate local work on your personal branch before switching branches
or pulling. If `--ff-only` reports divergent history, preserve your commits and
resolve the divergence together instead of force-pushing or resetting files.
The dataset and local Python environment must be supplied separately as above.

Do all work on a personal branch, then open a pull request for review:

```powershell
git checkout -b tejaswini/feature-name
git add src scripts tests README.md
git commit -m "Add preprocessing pipeline"
git push -u origin tejaswini/feature-name
```

## Project layout

```text
data/raw/          Original local datasets
data/processed/    Local cleaned datasets
notebooks/eda/     Exploratory data analysis notebooks
scripts/           Runnable commands
src/preprocessing/ Reusable cleaning and validation code
reports/metrics/   Local preprocessing and model reports
tests/             Automated checks for reusable code
```
