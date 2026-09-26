"""Tests for the Model 1 chronological baseline workflow."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaning import MODEL_1_OPERATIONAL_FEATURES
from src.training.train import train_and_evaluate_model_1_baseline


def test_model_1_baseline_uses_only_training_data_for_fitted_pipeline() -> None:
    rows = 40
    dataframe = pd.DataFrame(
        {
            feature: [row * (feature_index + 1) for row in range(rows)]
            for feature_index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES)
        }
    )
    dataframe["Timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="15min")
    dataframe["Failure_Probability"] = [int(row % 5 == 0) for row in range(rows)]
    dataframe["Maintenance_Type"] = 0
    dataframe["RUL"] = 100.0
    dataframe["TTF"] = 50.0
    dataframe["Component_Health_Score"] = 0.8
    original = dataframe.copy(deep=True)

    result = train_and_evaluate_model_1_baseline(dataframe)

    assert result.training_rows == 32
    assert result.testing_rows == 8
    assert result.input_features == MODEL_1_OPERATIONAL_FEATURES
    assert result.train_class_distribution == {0: 25, 1: 7}
    assert result.test_class_distribution == {0: 7, 1: 1}
    assert result.model_configuration["logistic_regression"]["class_weight"] == "balanced"
    assert result.majority_baseline_metrics["confusion_matrix"] == [[7, 0], [1, 0]]
    assert "preprocessor" in result.logistic_regression_pipeline.named_steps
    assert result.logistic_regression_pipeline.named_steps["classifier"].class_weight == "balanced"
    pd.testing.assert_frame_equal(dataframe, original)
