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
| Target selection | Complete: `Failure_Probability` |
| Model training and evaluation | Model 1 baseline complete |

The current dataset contains 175,393 records and 30 columns. The first
preprocessing run found no missing values and no exact duplicate rows.

## Setup

Create and activate a virtual environment, then install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Keep original datasets in `data/raw/`. Cleaned datasets are written to
`data/processed/`. Both folders are ignored by Git, so large data files are
kept local and never pushed to GitHub.

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

Run the baseline from the project root:

```powershell
python scripts/train_baseline.py
```

## Earlier planning notes

Model 2 will later treat `Maintenance_Type` as a multiclass target, but it has not started. `RUL`, `TTF`, and `Component_Health_Score` remain excluded from Model 1 because they may leak post-outcome or future information.

## Collaboration workflow

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
