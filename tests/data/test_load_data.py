"""Tests for dataset-loading helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.load_data import csv_has_title_row, load_dataset


def test_load_dataset_skips_a_title_row_before_csv_header(tmp_path: Path) -> None:
    dataset_path = tmp_path / "titled.csv"
    dataset_path.write_text("Dataset title\nTimestamp,SoC\n2024-01-01,0.8\n", encoding="utf-8")

    dataframe = load_dataset(dataset_path)

    assert csv_has_title_row(dataset_path) is True
    assert dataframe.columns.tolist() == ["Timestamp", "SoC"]
    assert dataframe.shape == (1, 2)


def test_load_dataset_keeps_a_regular_csv_header(tmp_path: Path) -> None:
    dataset_path = tmp_path / "regular.csv"
    dataset_path.write_text("Timestamp,SoC\n2024-01-01,0.8\n", encoding="utf-8")

    dataframe = load_dataset(dataset_path)

    assert csv_has_title_row(dataset_path) is False
    assert dataframe.columns.tolist() == ["Timestamp", "SoC"]
