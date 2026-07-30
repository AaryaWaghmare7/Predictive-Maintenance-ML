# Codex Project Notes

## Project Goal

Build a predictive maintenance machine learning model using the Kaggle dataset placed in `data/raw/`.

## Working Guidelines

- Keep raw dataset files in `data/raw/`.
- Save cleaned datasets in `data/processed/`.
- Put reusable Python code under `src/`.
- Put exploratory notebooks under `notebooks/eda/`.
- Put model experiments under `notebooks/modeling/`.
- Save trained model artifacts under `models/`.
- Save plots under `reports/figures/`.
- Save metrics under `reports/metrics/`.
- Do not commit large dataset files or model artifacts unless explicitly requested.

## Preferred Workflow

1. Inspect the dataset schema.
2. Clean and preprocess the data.
3. Build baseline classification/regression models.
4. Evaluate model performance.
5. Save the best model and document results.
