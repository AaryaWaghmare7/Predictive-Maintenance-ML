"""Utilities for loading predictive maintenance datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_raw_data_files(raw_dir: str | Path = "data/raw") -> list[Path]:
    """List common dataset files in the raw data directory."""
    raw_path = Path(raw_dir)
    patterns = ("*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_path.glob(pattern))
    return sorted(files)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset based on its file extension."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if suffix == ".json":
        return pd.read_json(dataset_path)

    raise ValueError(f"Unsupported dataset format: {suffix}")
