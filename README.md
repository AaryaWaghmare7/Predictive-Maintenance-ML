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
| Dataset inspection | Complete |
| Data preprocessing | Complete |
| Exploratory data analysis (EDA) | Complete |
| Target selection | Complete: Models 1 and 2 |
| Model 1 training and signal review | Logistic Regression baseline, Random Forest notebook experiment, and signal investigation complete; paused |
| Model 2 signal investigation | Complete; no classifier trained |

The current dataset contains 175,393 records and 30 columns. The first
preprocessing run found no missing values and no exact duplicate rows.

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

## Preprocessing

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

## Model 1 baseline

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
No additional training is part of repository synchronization.

For reference, the existing baseline entry point is:

```powershell
python scripts/train_baseline.py
```

## Model 2 status

Model 2 uses `Maintenance_Type`: 0 = None, 1 = Preventive, 2 = Corrective, and 3 = Predictive. Its signal investigation uses the same fixed 24 operational features as Model 1 and excludes `Timestamp`, `Failure_Probability`, `RUL`, `TTF`, and `Component_Health_Score` to avoid leakage. See `reports/07_Maintenance_Type_Signal_Analysis.md` before authorizing a multiclass baseline.

The investigation found extremely weak signal. No Model 2 classifier has been
trained. Dataset 2 has not started.

The completed Model 1 and Model 2 signal plots in `reports/figures/` are tracked
alongside their reports. Other generated data, metrics, and model artifacts
remain local and ignored by Git.

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
