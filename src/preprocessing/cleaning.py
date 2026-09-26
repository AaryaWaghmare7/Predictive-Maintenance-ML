"""Data cleaning helpers for EV motor datasets.

This file will contain small, reusable functions for cleaning raw
dataset columns before feature engineering or model training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_LEAKAGE_COLUMNS = (
    "Maintenance_Type",
    "RUL",
    "TTF",
    "Component_Health_Score",
)

MODEL_1_TARGET = "Failure_Probability"
MODEL_1_EXCLUDED_COLUMNS = (
    "Timestamp",
    MODEL_1_TARGET,
    *DEFAULT_LEAKAGE_COLUMNS,
)
MODEL_1_OPERATIONAL_FEATURES = (
    "SoC",
    "SoH",
    "Battery_Voltage",
    "Battery_Current",
    "Battery_Temperature",
    "Charge_Cycles",
    "Motor_Temperature",
    "Motor_Vibration",
    "Motor_Torque",
    "Motor_RPM",
    "Power_Consumption",
    "Brake_Pad_Wear",
    "Brake_Pressure",
    "Reg_Brake_Efficiency",
    "Tire_Pressure",
    "Tire_Temperature",
    "Suspension_Load",
    "Ambient_Temperature",
    "Ambient_Humidity",
    "Load_Weight",
    "Driving_Speed",
    "Distance_Traveled",
    "Idle_Time",
    "Route_Roughness",
)


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple snake_case column names.

    Example:
        "Motor Temperature (C)" becomes "motor_temperature_c".
    """
    cleaned = dataframe.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("[^a-z0-9_]", "", regex=True)
    )
    return cleaned


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with identical rows removed.

    Keeping the first occurrence makes the result predictable and avoids
    accidentally giving repeated observations extra weight later in a model.
    """
    return dataframe.drop_duplicates().copy()


def parse_and_sort_timestamp(
    dataframe: pd.DataFrame, timestamp_column: str = "timestamp"
) -> pd.DataFrame:
    """Convert a timestamp column to datetime and sort rows chronologically.

    The function leaves a dataset unchanged when it does not contain the
    requested column, so it can be reused with other EV datasets.
    """
    cleaned = dataframe.copy()
    if timestamp_column not in cleaned.columns:
        return cleaned

    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
    if cleaned[timestamp_column].isna().any():
        invalid_count = int(cleaned[timestamp_column].isna().sum())
        raise ValueError(
            f"{timestamp_column!r} contains {invalid_count} invalid timestamp value(s)."
        )
    return cleaned.sort_values(timestamp_column).reset_index(drop=True)


def replace_infinite_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with missing values."""
    cleaned = dataframe.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].replace([float("inf"), float("-inf")], pd.NA)
    return cleaned


def parse_timestamp_column(
    dataframe: pd.DataFrame, timestamp_column: str = "Timestamp"
) -> pd.DataFrame:
    """Return a copy with a validated datetime timestamp column.

    Invalid values raise an error rather than silently becoming missing data.
    The original column name is retained to keep the raw schema recognizable.
    """
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    cleaned = dataframe.copy()
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="raise")
    return cleaned


def select_classification_features(
    dataframe: pd.DataFrame,
    target_column: str = "Failure_Probability",
    timestamp_column: str = "Timestamp",
    leakage_columns: tuple[str, ...] = DEFAULT_LEAKAGE_COLUMNS,
) -> pd.DataFrame:
    """Return candidate sensor and operating features for failure classification.

    Timestamp stays in the dataset for ordering and audit checks, but is not an
    initial model input. Known target, post-outcome, and maintenance-result
    fields are excluded to prevent leakage. This function performs selection
    only; it does not fit any transformation.
    """
    exclusions = {target_column, timestamp_column, *leakage_columns}
    return dataframe.loc[:, [column for column in dataframe.columns if column not in exclusions]].copy()


def select_model_1_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed 24-column input matrix for Model 1.

    Model 1 classifies the documented ``Failure_Probability`` label. Selecting
    a fixed allowlist, rather than every column outside an exclusion list,
    prevents new or undocumented columns from silently entering the model.
    This function only selects columns; it does not fit or apply a transform.
    """
    missing_features = [
        feature for feature in MODEL_1_OPERATIONAL_FEATURES if feature not in dataframe.columns
    ]
    if missing_features:
        raise ValueError(f"Missing approved Model 1 features: {missing_features}")

    return dataframe.loc[:, list(MODEL_1_OPERATIONAL_FEATURES)].copy()


def get_model_1_xy(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return Model 1 inputs and the binary failure target without fitting data.

    The returned inputs contain only ``MODEL_1_OPERATIONAL_FEATURES``. The
    target is copied so callers cannot mutate the source dataframe by changing
    the returned values.
    """
    if MODEL_1_TARGET not in dataframe.columns:
        raise ValueError(f"Missing Model 1 target column: {MODEL_1_TARGET}")

    return select_model_1_features(dataframe), dataframe[MODEL_1_TARGET].copy()


def chronological_train_test_split(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered time-series data into earlier train and later test sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    ordered = dataframe.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    if split_index == 0 or split_index == len(ordered):
        raise ValueError("The dataset is too small for the requested split.")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def build_numeric_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Create an unfitted numeric preprocessing pipeline for later training.

    Median imputation and scaling are defined here, but fitting must happen only
    on the chronological training partition inside a future model pipeline.
    """
    if not numeric_features:
        raise ValueError("At least one numeric feature is required.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric_features)],
        remainder="drop",
    )
