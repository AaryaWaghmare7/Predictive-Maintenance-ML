"""Dataset validation helpers.

Validation checks help us catch common data problems early, such as
missing required columns or empty datasets.
"""

from __future__ import annotations

import pandas as pd


def ensure_not_empty(dataframe: pd.DataFrame) -> None:
    """Raise an error if the dataset has no rows."""
    if dataframe.empty:
        raise ValueError("Dataset is empty.")


def ensure_columns_exist(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise an error if any required columns are missing."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def ensure_binary_target(dataframe: pd.DataFrame, target_column: str) -> None:
    """Raise an error unless the target contains only binary 0/1 values."""
    ensure_columns_exist(dataframe, [target_column])
    if dataframe[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    values = set(dataframe[target_column].unique())
    if values != {0, 1}:
        raise ValueError(
            f"Target column '{target_column}' must contain both 0 and 1; found {sorted(values)}."
        )


def ensure_valid_timestamps(dataframe: pd.DataFrame, timestamp_column: str = "Timestamp") -> None:
    """Raise an error for missing, non-datetime, duplicate, or unsorted timestamps."""
    ensure_columns_exist(dataframe, [timestamp_column])
    timestamps = dataframe[timestamp_column]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise ValueError(f"Timestamp column '{timestamp_column}' must be datetime-like.")
    if timestamps.isna().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains missing values.")
    if timestamps.duplicated().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains duplicate values.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"Timestamp column '{timestamp_column}' must be sorted in ascending order.")


def dataset_quality_summary(
    dataframe: pd.DataFrame, timestamp_column: str = "Timestamp"
) -> dict[str, object]:
    """Return non-destructive data-quality measures for a preprocessing report."""
    summary: dict[str, object] = {
        "shape": dataframe.shape,
        "missing_values": dataframe.isna().sum().to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }
    if timestamp_column in dataframe.columns:
        timestamps = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
        summary.update(
            {
                "invalid_timestamps": int(timestamps.isna().sum()),
                "duplicate_timestamps": int(timestamps.duplicated().sum()),
                "timestamps_sorted": bool(timestamps.is_monotonic_increasing),
            }
        )
    return summary
