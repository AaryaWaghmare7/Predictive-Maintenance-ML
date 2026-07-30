"""Exploratory data analysis summary helpers.

EDA helps us understand the dataset before building any machine learning
model. These helpers are intentionally simple and beginner friendly.
"""

from __future__ import annotations

import pandas as pd


def dataset_overview(dataframe: pd.DataFrame) -> dict[str, object]:
    """Return basic information about rows, columns, and missing values."""
    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "missing_values": dataframe.isna().sum().to_dict(),
    }
