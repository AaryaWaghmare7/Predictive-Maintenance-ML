"""Tests for preprocessing validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.validation import (
    dataset_quality_summary,
    ensure_binary_target,
    ensure_valid_timestamps,
)


def test_ensure_binary_target_accepts_zero_and_one() -> None:
    ensure_binary_target(pd.DataFrame({"target": [0, 1, 0]}), "target")


def test_ensure_binary_target_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="must contain both 0 and 1"):
        ensure_binary_target(pd.DataFrame({"target": [0, 2]}), "target")


def test_ensure_binary_target_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="contains missing"):
        ensure_binary_target(pd.DataFrame({"target": [0, 1, None]}), "target")


def test_ensure_valid_timestamps_rejects_duplicates() -> None:
    dataframe = pd.DataFrame({"Timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"])})

    with pytest.raises(ValueError, match="duplicate"):
        ensure_valid_timestamps(dataframe)


def test_dataset_quality_summary_reports_non_destructive_findings() -> None:
    dataframe = pd.DataFrame(
        {"Timestamp": ["2024-01-01", "bad timestamp"], "value": [1, None]}
    )

    summary = dataset_quality_summary(dataframe)

    assert summary["shape"] == (2, 2)
    assert summary["missing_values"]["value"] == 1
    assert summary["invalid_timestamps"] == 1
