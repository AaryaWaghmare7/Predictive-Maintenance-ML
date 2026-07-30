# Predictive-Maintenance-ML

A machine learning-based predictive maintenance system that analyzes electric motor data to detect patterns, predict potential failures, and estimate failure risk before breakdowns occur.

## Project Structure

```text
Predictive-Maintenance-ML/
├── data/
│   ├── external/       # Third-party reference data
│   ├── interim/        # Intermediate transformed data
│   ├── raw/            # Original Kaggle dataset files
│   └── processed/      # Cleaned/transformed datasets
├── configs/            # Config files for training runs
├── docs/               # Project documentation
├── logs/               # Runtime logs
├── models/             # Trained model artifacts
├── notebooks/          # EDA and experiment notebooks
│   ├── eda/            # Exploratory analysis
│   └── modeling/       # Model experiments
├── outputs/
│   └── predictions/    # Prediction outputs
├── reports/
│   ├── figures/        # Plots and generated visuals
│   └── metrics/        # Evaluation metrics
├── scripts/            # Runnable project scripts
├── src/
│   ├── data/           # Dataset loading utilities
│   ├── evaluation/     # Metrics and validation helpers
│   ├── features/       # Feature engineering code
│   ├── models/         # Training and prediction code
│   ├── utils/          # Shared helpers
│   └── visualization/  # Plotting utilities
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

To inspect the first dataset file found in `data/raw/`, run:

```bash
python scripts/inspect_dataset.py
```

After choosing the target column, train a baseline model with:

```bash
python scripts/train_baseline.py --data data/raw/your_dataset.csv --target your_target_column
```

## Next Steps

1. Add the Kaggle dataset to `data/raw/`.
2. Create an exploratory notebook in `notebooks/`.
3. Build preprocessing code in `src/features/`.
4. Train baseline models in `src/models/`.
