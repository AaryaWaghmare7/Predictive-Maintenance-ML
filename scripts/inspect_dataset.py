"""Inspect the first dataset found in data/raw."""

from __future__ import annotations

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
