# EV Predictive Maintenance AI

A beginner-friendly machine-learning project for predictive maintenance of electric vehicle motors using publicly available electric motor/EV datasets.

Tesla is used only as an industry case study. This project does not use Tesla proprietary data.

## Project Structure

```text
EV-Predictive-Maintenance-AI/
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
│   ├── api/            # Future web/API deployment
│   ├── config/         # Shared project paths/settings
│   ├── data/           # Dataset loading utilities
│   ├── eda/            # Exploratory data analysis helpers
│   ├── evaluation/     # Metrics and validation helpers
│   ├── features/       # Feature engineering code
│   ├── models/         # Model-related code
│   ├── prediction/     # Future prediction pipeline
│   ├── preprocessing/  # Data cleaning and validation
│   ├── storage/        # Model storage helpers
│   ├── training/       # Future training entry points
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

Model training is intentionally not implemented yet.

After the dataset is inspected and the target column is chosen, training code can be added under `src/training/`.

The placeholder training script currently prints a message:

```bash
python scripts/train_baseline.py
```

## Next Steps

1. Add the Kaggle dataset to `data/raw/`.
2. Run `python scripts/inspect_dataset.py`.
3. Build preprocessing code in `src/features/`.
4. Create an exploratory notebook in `notebooks/eda/`.
5. Decide the target column before any model training.

See `docs/PROJECT_STRUCTURE.md` for a folder-by-folder explanation.
