"""Inspect the first dataset found in data/raw."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow the documented ``python scripts/inspect_dataset.py`` command to find
# project packages without requiring an absolute path or package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import list_raw_data_files, load_dataset


def main() -> None:
    files = list_raw_data_files()
    if not files:
        print("No dataset files found in data/raw.")
        return

    dataset_path = files[0]
    dataframe = load_dataset(dataset_path)

    print(f"Loaded: {dataset_path}")
    print(f"Shape: {dataframe.shape}")
    print("\nColumns:")
    print(dataframe.columns.tolist())
    print("\nPreview:")
    print(dataframe.head())
    print("\nMissing values:")
    print(dataframe.isna().sum())


if __name__ == "__main__":
    main()
