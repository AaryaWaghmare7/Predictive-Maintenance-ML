"""Training workflow for the first Model 1 baseline.

This module contains only the approved Logistic Regression baseline for
``Failure_Probability``. It does not perform feature engineering, resampling,
threshold tuning, or hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.paths import RAW_DATA_DIR
from src.data.load_data import load_dataset
from src.evaluation.metrics import binary_classification_metrics
from src.preprocessing.cleaning import (
    MODEL_1_OPERATIONAL_FEATURES,
    build_numeric_preprocessor,
    chronological_train_test_split,
    get_model_1_xy,
    parse_timestamp_column,
)
from src.preprocessing.validation import ensure_binary_target, ensure_valid_timestamps


@dataclass
class Model1BaselineResult:
    """Models, split details, and test-only results from one baseline run."""

    training_rows: int
    testing_rows: int
    input_features: tuple[str, ...]
    train_class_distribution: dict[int, int]
    test_class_distribution: dict[int, int]
    model_configuration: dict[str, Any]
    majority_baseline_metrics: dict[str, Any]
    logistic_regression_metrics: dict[str, Any]
    majority_baseline: DummyClassifier
    logistic_regression_pipeline: Pipeline


def _class_distribution(target: pd.Series) -> dict[int, int]:
    """Return sorted integer class counts for concise reporting."""
    return {int(label): int(count) for label, count in target.value_counts().sort_index().items()}


def build_balanced_logistic_regression_pipeline(
    numeric_features: list[str],
) -> Pipeline:
    """Create an unfitted preprocessing-and-Logistic-Regression pipeline.

    The preprocessor is intentionally inside the pipeline, so imputation and
    scaling are learned only when the pipeline is fit on the training data.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_numeric_preprocessor(numeric_features)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_and_evaluate_model_1_baseline(dataframe: pd.DataFrame) -> Model1BaselineResult:
    """Train Model 1 baselines on earlier data and evaluate once on later data.

    The function uses the fixed 24-feature allowlist and never shuffles rows.
    Both models are fit on the training partition only; all returned metrics
    are calculated solely from the untouched chronological test partition.
    """
    data = parse_timestamp_column(dataframe)
    ensure_binary_target(data, "Failure_Probability")
    data = data.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    ensure_valid_timestamps(data)

    train_data, test_data = chronological_train_test_split(data, test_size=0.2)
    x_train, y_train = get_model_1_xy(train_data)
    x_test, y_test = get_model_1_xy(test_data)

    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(x_train, y_train)
    majority_predictions = majority_baseline.predict(x_test)
    majority_scores = majority_baseline.predict_proba(x_test)[:, 1]

    logistic_pipeline = build_balanced_logistic_regression_pipeline(x_train.columns.tolist())
    logistic_pipeline.fit(x_train, y_train)
    logistic_predictions = logistic_pipeline.predict(x_test)
    logistic_scores = logistic_pipeline.predict_proba(x_test)[:, 1]

    configuration = {
        "split_strategy": "chronological earliest 80% train / latest 20% test",
        "feature_count": len(MODEL_1_OPERATIONAL_FEATURES),
        "majority_baseline": {"estimator": "DummyClassifier", "strategy": "most_frequent"},
        "logistic_regression": {
            "estimator": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "preprocessing": "median imputation followed by StandardScaler inside Pipeline",
        },
    }

    return Model1BaselineResult(
        training_rows=len(train_data),
        testing_rows=len(test_data),
        input_features=tuple(x_train.columns),
        train_class_distribution=_class_distribution(y_train),
        test_class_distribution=_class_distribution(y_test),
        model_configuration=configuration,
        majority_baseline_metrics=binary_classification_metrics(
            y_test, majority_predictions, majority_scores
        ),
        logistic_regression_metrics=binary_classification_metrics(
            y_test, logistic_predictions, logistic_scores
        ),
        majority_baseline=majority_baseline,
        logistic_regression_pipeline=logistic_pipeline,
    )


def load_and_run_model_1_baseline(
    dataset_path: str | Path = RAW_DATA_DIR / "EV_Predictive_Maintenance_Dataset_15min.csv",
) -> Model1BaselineResult:
    """Load the raw dataset and run the approved Model 1 baseline workflow."""
    return train_and_evaluate_model_1_baseline(load_dataset(dataset_path))

def main() -> None:
    """Run and print the test-only results for the approved baseline."""
    result = load_and_run_model_1_baseline()
    print(f"Training rows: {result.training_rows:,}")
    print(f"Testing rows: {result.testing_rows:,}")
    print(f"Input features: {len(result.input_features)}")
    print(f"Train classes: {result.train_class_distribution}")
    print(f"Test classes: {result.test_class_distribution}")
    print("\nMajority baseline metrics:")
    print(result.majority_baseline_metrics)
    print("\nBalanced Logistic Regression metrics:")
    print(result.logistic_regression_metrics)


if __name__ == "__main__":
    main()
