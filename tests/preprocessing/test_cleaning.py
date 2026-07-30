"""Tests for preprocessing helper functions."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaning import normalize_column_names


def test_normalize_column_names() -> None:
    dataframe = pd.DataFrame(columns=[" Motor Temperature (C) ", "Vehicle Speed"])

    cleaned = normalize_column_names(dataframe)

    assert cleaned.columns.tolist() == ["motor_temperature_c", "vehicle_speed"]
