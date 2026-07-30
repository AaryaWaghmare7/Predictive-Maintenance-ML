# Codex Project Notes

## Project Goal

Build "EV Predictive Maintenance AI", a beginner-friendly predictive maintenance project for electric vehicle motors using publicly available electric motor/EV datasets.

Tesla may be discussed only as an industry case study. This project does not use Tesla proprietary data.

## Working Guidelines

- Keep code beginner friendly, clear, and well documented.
- Do not build the ML model until the dataset has been inspected and the user asks to proceed.
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
3. Perform exploratory data analysis.
4. Create feature engineering functions.
5. Build baseline classification/regression models only after approval.
6. Evaluate model performance.
7. Save the best model and document results.
