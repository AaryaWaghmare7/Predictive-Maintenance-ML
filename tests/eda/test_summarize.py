"""Tests for EDA summary helpers."""

from __future__ import annotations

import pandas as pd

from src.eda.summarize import dataset_overview


def test_dataset_overview() -> None:
    dataframe = pd.DataFrame({"temperature": [30, None], "current": [10, 12]})

    overview = dataset_overview(dataframe)

    assert overview["rows"] == 2
    assert overview["columns"] == 2
    assert overview["missing_values"]["temperature"] == 1
