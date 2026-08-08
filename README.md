# EV Predictive Maintenance AI

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
| Exploratory data analysis (EDA) | Next |
| Target selection | Pending |
| Model training and evaluation | Pending |

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

## Next step: EDA and target selection

Create an exploratory notebook in `notebooks/eda/` using the cleaned data.
The notebook should examine distributions, sensor relationships, and the
possible prediction targets:

- `failure_probability`
- `maintenance_type`
- `rul` (remaining useful life)
- `ttf` (time to failure)
- `component_health_score`

Choose one target before training. For example, predicting `rul` is a
regression task, while predicting `maintenance_type` is a classification task.

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
