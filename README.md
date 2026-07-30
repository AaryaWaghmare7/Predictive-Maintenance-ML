# Predictive-Maintenance-ML

A machine learning-based predictive maintenance system that analyzes electric motor data to detect patterns, predict potential failures, and estimate failure risk before breakdowns occur.

## Project Structure

```text
Predictive-Maintenance-ML/
├── data/
│   ├── raw/            # Original Kaggle dataset files
│   └── processed/      # Cleaned/transformed datasets
├── notebooks/          # EDA and experiment notebooks
├── reports/
│   └── figures/        # Plots and generated visuals
├── src/
│   ├── features/       # Feature engineering code
│   └── models/         # Training and prediction code
└── tests/              # Tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the downloaded Kaggle dataset files in `data/raw/`.

Large data files are intentionally ignored by Git. Keep only small placeholder files such as `.gitkeep` in the repository.

## Next Steps

1. Add the Kaggle dataset to `data/raw/`.
2. Create an exploratory notebook in `notebooks/`.
3. Build preprocessing code in `src/features/`.
4. Train baseline models in `src/models/`.
