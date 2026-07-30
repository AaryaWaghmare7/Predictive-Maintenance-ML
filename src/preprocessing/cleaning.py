"""Data cleaning helpers for EV motor datasets.

This file will contain small, reusable functions for cleaning raw
dataset columns before feature engineering or model training.
"""

from __future__ import annotations

import pandas as pd


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple snake_case column names.

    Example:
        "Motor Temperature (C)" becomes "motor_temperature_c".
    """
    cleaned = dataframe.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("[^a-z0-9_]", "", regex=True)
    )
    return cleaned
