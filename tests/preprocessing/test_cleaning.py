"""Tests for preprocessing helper functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.cleaning import (
    DEFAULT_LEAKAGE_COLUMNS,
    MODEL_1_EXCLUDED_COLUMNS,
    MODEL_1_OPERATIONAL_FEATURES,
    MODEL_1_TARGET,
    build_numeric_preprocessor,
    chronological_train_test_split,
    get_model_1_xy,
    normalize_column_names,
    parse_timestamp_column,
    select_classification_features,
    select_model_1_features,
)


def test_normalize_column_names() -> None:
    dataframe = pd.DataFrame(columns=[" Motor Temperature (C) ", "Vehicle Speed"])

    cleaned = normalize_column_names(dataframe)

    assert cleaned.columns.tolist() == ["motor_temperature_c", "vehicle_speed"]


def test_parse_timestamp_column_preserves_name_and_parses_values() -> None:
    dataframe = pd.DataFrame({"Timestamp": ["2024-01-01 00:00:00"]})

    cleaned = parse_timestamp_column(dataframe)

    assert str(cleaned["Timestamp"].dtype).startswith("datetime64")
    assert dataframe["Timestamp"].dtype == object


def test_select_classification_features_excludes_target_time_and_leakage() -> None:
    dataframe = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-01"]),
            "SoC": [0.8],
            "Failure_Probability": [0],
            "Maintenance_Type": [1],
            "RUL": [100],
            "TTF": [80],
            "Component_Health_Score": [0.9],
        }
    )

    features = select_classification_features(dataframe)

    assert features.columns.tolist() == ["SoC"]
    assert set(DEFAULT_LEAKAGE_COLUMNS).isdisjoint(features.columns)


def test_chronological_train_test_split_keeps_later_data_for_test() -> None:
    dataframe = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-04", "2024-01-02"]),
            "SoC": [3, 1, 4, 2],
        }
    )

    train, test = chronological_train_test_split(dataframe, test_size=0.5)

    assert train["Timestamp"].max() < test["Timestamp"].min()
    assert train["SoC"].tolist() == [1, 2]


def test_build_numeric_preprocessor_is_unfitted_until_training() -> None:
    preprocessor = build_numeric_preprocessor(["SoC"])

    with pytest.raises(Exception):
        preprocessor.transform(pd.DataFrame({"SoC": [0.5]}))


def test_model_1_selection_uses_the_exact_approved_feature_allowlist() -> None:
    dataframe = pd.DataFrame(
        {
            **{feature: [index] for index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES)},
            "Timestamp": pd.to_datetime(["2024-01-01"]),
            MODEL_1_TARGET: [1],
            "Maintenance_Type": [2],
            "RUL": [50],
            "TTF": [30],
            "Component_Health_Score": [0.7],
            "Undocumented_Future_Column": [999],
        }
    )

    features, target = get_model_1_xy(dataframe)

    assert features.columns.tolist() == list(MODEL_1_OPERATIONAL_FEATURES)
    assert features.shape[1] == 24
    assert MODEL_1_TARGET not in features
    assert set(MODEL_1_EXCLUDED_COLUMNS).isdisjoint(features.columns)
    assert "Undocumented_Future_Column" not in features
    assert target.tolist() == [1]


def test_model_1_selection_does_not_mutate_source_data() -> None:
    dataframe = pd.DataFrame(
        {
            **{feature: [0.5, 0.8] for feature in MODEL_1_OPERATIONAL_FEATURES},
            "Timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:15:00"]),
            MODEL_1_TARGET: [0, 1],
            "Maintenance_Type": [0, 1],
            "RUL": [100, 99],
            "TTF": [80, 79],
            "Component_Health_Score": [0.9, 0.8],
        }
    )
    original = dataframe.copy(deep=True)

    features = select_model_1_features(dataframe)
    features.iloc[0, 0] = -1

    pd.testing.assert_frame_equal(dataframe, original)
