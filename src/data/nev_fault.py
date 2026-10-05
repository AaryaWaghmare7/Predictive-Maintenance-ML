"""Experiment 2 schema and supplied split checks; never rewrite raw CSVs."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import RAW_DATA_DIR
from src.data.load_data import load_dataset

NEV_DATA_DIR = RAW_DATA_DIR / "experiment_2"
NEV_FILES = {
    "full": "NEV_fault_dataset.csv",
    "training": "NEV_fault_training_dataset.csv",
    "testing": "NEV_fault_testing_dataset.csv",
}
NEV_FEATURES = (
    "Voltage (V)", "Current (A)", "Motor Speed (RPM)",
    "Temperature (°C)", "Vibration (g)", "Ambient Temp (°C)", "Humidity (%)",
)
NEV_TARGET = "Fault Label"
NEV_CLASSES = {0: "Normal", 1: "Motor Fault", 2: "Inverter Fault", 3: "Battery Fault"}
NORMALIZED_RANGE_TOLERANCE = 1e-12


def file_sha256(path: str | Path) -> str:
    """Fingerprint original bytes for preservation and reproducibility checks."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_nev_frame(frame: pd.DataFrame, *, require_target: bool = True) -> None:
    """Reject incompatible schema, invalid labels and unsupported input ranges.

    Names retain their original physical-unit text, but values are normalized.
    No row removal, imputation, clipping or column renaming happens here.
    """
    expected = set(NEV_FEATURES) | ({NEV_TARGET} if require_target else set())
    if frame.columns.has_duplicates or set(frame.columns) != expected:
        raise ValueError(f"NEV schema must contain exactly {sorted(expected)}")
    if frame.empty:
        raise ValueError("NEV data must not be empty")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns):
        raise ValueError("NEV columns must be numeric")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NEV data contains missing or infinite values; investigate first")
    sensors = frame.loc[:, list(NEV_FEATURES)]
    # Original files contain 1.0000000000000002: a floating-point boundary
    # excess of 2.22e-16. Accept numerical roundoff without changing any value.
    if ((sensors < -NORMALIZED_RANGE_TOLERANCE) | (sensors > 1 + NORMALIZED_RANGE_TOLERANCE)).any().any():
        raise ValueError("NEV sensor inputs must be normalized to [0, 1] within 1e-12; never silently clip")
    if require_target and not frame[NEV_TARGET].isin(NEV_CLASSES).all():
        raise ValueError("Fault Label must contain integer-valued codes 0, 1, 2, 3")


def get_nev_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return seven named inputs in canonical order and an integer label copy."""
    validate_nev_frame(frame)
    return frame.loc[:, list(NEV_FEATURES)].copy(), frame[NEV_TARGET].astype(int).copy()


def verify_supplied_split(
    full: pd.DataFrame, training: pd.DataFrame, testing: pd.DataFrame
) -> dict[str, object]:
    """Check exact multiset membership, multiplicity, and feature-only overlap.

    This verifies supplied membership, not temporal or cross-vehicle separation.
    File/row order is not treated as an identifier or feature.
    """
    for frame in (full, training, testing):
        validate_nev_frame(frame)
    columns = [*NEV_FEATURES, NEV_TARGET]
    rows = [Counter(frame.loc[:, columns].itertuples(index=False, name=None))
            for frame in (full, training, testing)]
    feature_rows = [set(frame.loc[:, list(NEV_FEATURES)].itertuples(index=False, name=None))
                    for frame in (training, testing)]
    result = {
        "training_plus_testing_equals_full_multiset": rows[0] == rows[1] + rows[2],
        "exact_row_overlap": len(rows[1].keys() & rows[2].keys()),
        "feature_only_overlap": len(feature_rows[0] & feature_rows[1]),
        "compatible_schema": True,
    }
    if not result["training_plus_testing_equals_full_multiset"]:
        raise ValueError("Supplied train/test rows do not exactly reconstruct the full dataset")
    if result["feature_only_overlap"]:
        raise ValueError("Training and testing contain overlapping input observations")
    return result


def load_nev_datasets(data_dir: str | Path = NEV_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Use the existing loader and preserve the supplied training/testing files."""
    frames = {name: load_dataset(Path(data_dir) / filename) for name, filename in NEV_FILES.items()}
    verify_supplied_split(frames["full"], frames["training"], frames["testing"])
    return frames
