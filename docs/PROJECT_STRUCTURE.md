# Project Structure

The existing repository supports two preserved ML experiments and a localhost
application. Experiment 1 is retained as a weak-signal experiment; Experiment 2
provides proof-of-concept NEV fault diagnosis. RUL has contracts only.

## Root Files

- `README.md` explains the project goal, setup, and how to use the repository.
- `PROJECT_PLAN.md` and `DECISIONS.md` record scope, evidence and approval boundaries.
- `AGENTS.md` gives Codex project-specific instructions.
- `requirements.txt` lists Python packages needed for development.
- `.gitignore` excludes datasets, other model artifacts, caches and private
  settings, with an explicit exception for the small approved NEV inference pipeline.

## Configuration

- `configs/model_config.yaml` stores future training settings such as data locations, target column, test size, and output folders.
- `src/config/paths.py` defines important project paths in one reusable Python file.

## Data Folders

- `data/raw/` stores original Kaggle or public EV/electric motor datasets.
- `data/external/` stores third-party reference data that is not the main dataset.
- `data/interim/` stores temporary intermediate files created during preprocessing.
- `data/processed/` stores cleaned datasets ready for modeling.
- `data/rul/` reserves a location for a manually approved degradation dataset;
  none is selected.

Large data files are ignored by Git so the repository stays lightweight.

## Notebook Folders

- `notebooks/eda/` is for exploratory data analysis notebooks.
- `notebooks/preprocessing/` documents preprocessing.
- `notebooks/modeling/` retains model experiments and their recorded outputs.

## Source Code Folders

- `src/data/` loads datasets from CSV, Excel, Parquet, or JSON files.
- `src/preprocessing/` cleans and validates raw data.
- `src/eda/` creates summaries that help us understand the dataset.
- `src/features/` will contain feature engineering code.
- `src/training/` implements the existing baseline and NEV training workflows;
  the application never calls them.
- `src/evaluation/` contains metrics and model-quality checks.
- `src/storage/` defines where trained model files will be saved and loaded.
- `src/prediction/` performs shared saved-model inference in canonical feature order.
- `src/api/` contains FastAPI endpoints, strict request/response schemas and CSV validation.
- `src/api/static/` contains the plain HTML/CSS/JavaScript localhost dashboard.
- `src/rul/` contains future component-model contracts, not a fitted RUL model.
- `src/utils/` is for shared helper functions.
- `src/visualization/` is for reusable plotting functions.

## Script Folders

- `scripts/inspect_dataset.py` prints dataset shape, columns, preview rows, and missing values.
- `scripts/train_baseline.py` is the completed Experiment 1 baseline entry point.
- `scripts/analyze_failure_signal.py` and `scripts/analyze_maintenance_type_signal.py`
  run the existing Experiment 1 investigations.
- `scripts/run_nev_experiment.py` reproduces Experiment 2, including training;
  **do not run it merely to start the application**.
- `scripts/smoke_test_nev_inference.py` verifies the existing saved model
  without fitting anything.
- `scripts/smoke_test_localhost.py` checks the running API/dashboard assets,
  all four supplied examples, invalid inputs and CSV upload over HTTP.

## Output Folders

- `models/experiment_2/` stores the intentionally tracked NEV inference pipeline,
  unchanged model metadata and its artifact/checksum documentation.
- `models/rul/` reserves space for future RUL artifacts; none is trained.
- `reports/figures/` stores EDA and model plots.
- `reports/metrics/` stores model evaluation/statistical snapshots.
- `reports/11_RUL_Dataset_Requirements.md` records manual dataset-approval requirements.
- `reports/rul/` reserves space for future RUL evidence.
- `outputs/predictions/` will store future prediction files.
- `logs/` will store runtime logs if needed.

## Testing Folders

- `tests/data/` is for dataset-loading tests.
- `tests/preprocessing/` is for cleaning and validation tests.
- `tests/features/` is for feature engineering tests.
- `tests/models/` is for model tests.
- `tests/evaluation/` is for metric tests.
- `tests/prediction/` is for prediction tests.
- `tests/api/` checks health, prediction validation, CSV upload and dashboard assets.

API fixtures fit tiny synthetic classifiers in temporary directories; they do
not regenerate the production NEV artifact. Integration tests require the
tracked NEV pipeline and matching dependency versions. Raw data and other
fitted binaries stay Git-ignored. See `MODULAR_ARCHITECTURE.md` for model-routing
and coverage limits.
