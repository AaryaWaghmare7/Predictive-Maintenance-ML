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
