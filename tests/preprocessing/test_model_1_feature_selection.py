"""Integration checks for the locked Model 1 feature-selection policy."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from src.data.load_data import load_dataset
from src.preprocessing.cleaning import (
    MODEL_1_EXCLUDED_COLUMNS,
    MODEL_1_OPERATIONAL_FEATURES,
    MODEL_1_TARGET,
    chronological_train_test_split,
    get_model_1_xy,
    parse_timestamp_column,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = PROJECT_ROOT / "data/raw/EV_Predictive_Maintenance_Dataset_15min.csv"


@pytest.mark.skipif(not RAW_DATASET_PATH.exists(), reason="Raw dataset is not available locally.")
def test_model_1_selection_keeps_the_raw_csv_unchanged() -> None:
    """Load and select from the real CSV without modifying its bytes."""
    before_hash = sha256(RAW_DATASET_PATH.read_bytes()).hexdigest()

    dataframe = parse_timestamp_column(load_dataset(RAW_DATASET_PATH))
    train_data, test_data = chronological_train_test_split(dataframe)
    train_features, train_target = get_model_1_xy(train_data)
    test_features, test_target = get_model_1_xy(test_data)

    after_hash = sha256(RAW_DATASET_PATH.read_bytes()).hexdigest()

    assert before_hash == after_hash
    assert train_features.columns.tolist() == list(MODEL_1_OPERATIONAL_FEATURES)
    assert test_features.columns.tolist() == list(MODEL_1_OPERATIONAL_FEATURES)
    assert train_features.shape == (140_314, 24)
    assert test_features.shape == (35_079, 24)
    assert MODEL_1_TARGET not in train_features
    assert set(MODEL_1_EXCLUDED_COLUMNS).isdisjoint(train_features.columns)
    assert train_target.name == MODEL_1_TARGET
    assert test_target.name == MODEL_1_TARGET
    assert train_data["Timestamp"].max() < test_data["Timestamp"].min()
