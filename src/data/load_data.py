"""Utilities for loading predictive maintenance datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def list_raw_data_files(raw_dir: str | Path = "data/raw") -> list[Path]:
    """List common dataset files in the raw data directory."""
    raw_path = Path(raw_dir)
    patterns = ("*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_path.glob(pattern))
    return sorted(files)


def csv_has_title_row(path: str | Path) -> bool:
    """Return whether a CSV starts with a title line before its header.

    The supplied EV dataset starts with a one-cell title line followed by
    a comma-separated header. Comparing the number of delimiters in the
    first two non-empty lines lets us support that format without tying the
    loader to a particular file name or machine-specific path.
    """
    with Path(path).open(encoding="utf-8-sig") as file_handle:
        non_empty_lines: list[str] = []
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line:
                non_empty_lines.append(stripped_line)
            if len(non_empty_lines) == 2:
                break

    if len(non_empty_lines) < 2:
        return False

    return non_empty_lines[0].count(",") < non_empty_lines[1].count(",")


def load_dataset(path: str | Path, **read_options: Any) -> pd.DataFrame:
    """Load a dataset based on its file extension.

    CSV files with a title row before their header are detected and loaded
    with ``skiprows=1``. Callers may still provide normal pandas read options.
    """
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        csv_options = {"low_memory": False, **read_options}
        if "skiprows" not in csv_options and csv_has_title_row(dataset_path):
            csv_options["skiprows"] = 1
        return pd.read_csv(dataset_path, **csv_options)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if suffix == ".json":
        return pd.read_json(dataset_path)

    raise ValueError(f"Unsupported dataset format: {suffix}")
