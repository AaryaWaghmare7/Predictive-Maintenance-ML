"""Clean the EV predictive-maintenance dataset and save reusable outputs.

Example:
    python scripts/preprocess_dataset.py

By default, this script reads the first supported file in ``data/raw``. Use
``--input`` to run it against a dataset stored elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Allow ``python scripts/preprocess_dataset.py`` to import code from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.load_data import list_raw_data_files, load_dataset
from src.preprocessing.cleaning import (
    normalize_column_names,
    parse_and_sort_timestamp,
    remove_duplicate_rows,
    replace_infinite_values,
)
from src.preprocessing.validation import ensure_not_empty


def preprocess(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply safe, documented preprocessing without inventing data values."""
    ensure_not_empty(dataframe)
    original_rows, original_columns = dataframe.shape

    cleaned = normalize_column_names(dataframe)
    cleaned = replace_infinite_values(cleaned)
    cleaned = remove_duplicate_rows(cleaned)
    cleaned = parse_and_sort_timestamp(cleaned)

    missing_values = cleaned.isna().sum()
    remaining_missing = {
        column: int(count) for column, count in missing_values.items() if count > 0
    }
    if remaining_missing:
        raise ValueError(
            "Missing values remain after cleaning. Decide an imputation strategy "
            f"before modelling: {remaining_missing}"
        )

    report: dict[str, object] = {
        "original_shape": {"rows": original_rows, "columns": original_columns},
        "processed_shape": {"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1])},
        "duplicates_removed": int(original_rows - cleaned.shape[0]),
        "missing_values_after_cleaning": remaining_missing,
        "column_names": cleaned.columns.tolist(),
        "data_types": {column: str(dtype) for column, dtype in cleaned.dtypes.items()},
    }
    return cleaned, report


def find_input_file(input_path: str | None) -> Path:
    """Return the explicitly requested input file or the first raw dataset."""
    if input_path:
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input dataset was not found: {path}")
        return path

    files = list_raw_data_files(RAW_DATA_DIR)
    if not files:
        raise FileNotFoundError(
            "No dataset found in data/raw. Copy a CSV or Excel file there, or use --input."
        )
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a CSV, XLSX, XLS, Parquet, or JSON dataset.")
    parser.add_argument(
        "--output",
        default=PROCESSED_DATA_DIR / "ev_predictive_maintenance_cleaned.csv",
        type=Path,
        help="Destination CSV path.",
    )
    args = parser.parse_args()

    input_path = find_input_file(args.input)
    processed, report = preprocess(load_dataset(input_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)

    report_path = METRICS_DIR / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Processed dataset saved to: {args.output}")
    print(f"Preprocessing report saved to: {report_path}")
    print(f"Rows: {report['original_shape']['rows']} -> {report['processed_shape']['rows']}")


if __name__ == "__main__":
    main()
