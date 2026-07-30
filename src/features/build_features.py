"""Feature engineering helpers for EV predictive maintenance data.

This file keeps beginner-friendly feature utilities. No real model
features are created yet because the dataset schema is still unknown.
"""

from __future__ import annotations

import pandas as pd


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
