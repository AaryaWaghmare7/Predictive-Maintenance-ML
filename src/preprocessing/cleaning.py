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


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with identical rows removed.

    Keeping the first occurrence makes the result predictable and avoids
    accidentally giving repeated observations extra weight later in a model.
    """
    return dataframe.drop_duplicates().copy()


def parse_and_sort_timestamp(
    dataframe: pd.DataFrame, timestamp_column: str = "timestamp"
) -> pd.DataFrame:
    """Convert a timestamp column to datetime and sort rows chronologically.

    The function leaves a dataset unchanged when it does not contain the
    requested column, so it can be reused with other EV datasets.
    """
    cleaned = dataframe.copy()
    if timestamp_column not in cleaned.columns:
        return cleaned

    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
    if cleaned[timestamp_column].isna().any():
        invalid_count = int(cleaned[timestamp_column].isna().sum())
        raise ValueError(
            f"{timestamp_column!r} contains {invalid_count} invalid timestamp value(s)."
        )
    return cleaned.sort_values(timestamp_column).reset_index(drop=True)


def replace_infinite_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with missing values."""
    cleaned = dataframe.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].replace([float("inf"), float("-inf")], pd.NA)
    return cleaned
