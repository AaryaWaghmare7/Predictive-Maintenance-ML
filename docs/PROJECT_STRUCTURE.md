# Project Structure

This repository is organized for a beginner-friendly machine-learning workflow.
The current goal is only to prepare the structure. The model is not built yet.

## Root Files

- `README.md` explains the project goal, setup, and how to use the repository.
- `AGENTS.md` gives Codex project-specific instructions.
- `requirements.txt` lists Python packages needed for development.
- `.gitignore` prevents large datasets, model artifacts, caches, and local settings from being committed.

## Configuration

- `configs/model_config.yaml` stores future training settings such as data locations, target column, test size, and output folders.
- `src/config/paths.py` defines important project paths in one reusable Python file.

## Data Folders

- `data/raw/` stores original Kaggle or public EV/electric motor datasets.
- `data/external/` stores third-party reference data that is not the main dataset.
- `data/interim/` stores temporary intermediate files created during preprocessing.
- `data/processed/` stores cleaned datasets ready for modeling.

Large data files are ignored by Git so the repository stays lightweight.

## Notebook Folders

- `notebooks/eda/` is for exploratory data analysis notebooks.
- `notebooks/modeling/` is for future model experiments after the dataset is understood.

## Source Code Folders

- `src/data/` loads datasets from CSV, Excel, Parquet, or JSON files.
- `src/preprocessing/` cleans and validates raw data.
- `src/eda/` creates summaries that help us understand the dataset.
- `src/features/` will contain feature engineering code.
- `src/training/` is reserved for future training scripts.
- `src/evaluation/` will contain accuracy, error, and model-quality checks.
- `src/storage/` defines where trained model files will be saved and loaded.
- `src/prediction/` is reserved for future prediction code.
- `src/api/` is reserved for eventual web/API deployment.
- `src/utils/` is for shared helper functions.
- `src/visualization/` is for reusable plotting functions.

## Script Folders

- `scripts/inspect_dataset.py` prints dataset shape, columns, preview rows, and missing values.
- `scripts/train_baseline.py` is only a placeholder right now. It does not train a model yet.

## Output Folders

- `models/` will store trained model artifacts later.
- `reports/figures/` will store EDA and model plots.
- `reports/metrics/` will store model evaluation results later.
- `outputs/predictions/` will store future prediction files.
- `logs/` will store runtime logs if needed.

## Testing Folders

- `tests/data/` is for dataset-loading tests.
- `tests/preprocessing/` is for cleaning and validation tests.
- `tests/features/` is for feature engineering tests.
- `tests/models/` is for future model tests.
- `tests/evaluation/` is for metric tests.
- `tests/prediction/` is for prediction tests.
- `tests/api/` is for future API tests.
