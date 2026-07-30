"""Feature engineering helpers for predictive maintenance data."""

from __future__ import annotations

import pandas as pd


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load a raw CSV dataset from disk."""
    return pd.read_csv(path)


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    cleaned = dataframe.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("[^a-z0-9_]", "", regex=True)
    )
    return cleaned
