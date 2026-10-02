"""Leakage-safe signal diagnostics for the Model 2 maintenance-type target.

These helpers describe the association between the fixed operational feature
allowlist and the four documented ``Maintenance_Type`` classes. They do not
train a classifier, fit a preprocessing transformation, or alter raw data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from src.preprocessing.cleaning import (
    MODEL_1_OPERATIONAL_FEATURES,
    MODEL_2_CLASS_LABELS,
    MODEL_2_EXCLUDED_COLUMNS,
    MODEL_2_TARGET,
    chronological_train_test_split,
    get_model_2_xy,
    parse_timestamp_column,
)
from src.preprocessing.validation import ensure_target_classes, ensure_valid_timestamps


MAINTENANCE_CLASSES = tuple(MODEL_2_CLASS_LABELS)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class MaintenanceTypeSignalAnalysis:
    """Descriptive results from a chronology-respecting multiclass investigation."""

    class_distribution: pd.DataFrame
    feature_summary: pd.DataFrame
    class_feature_summary: pd.DataFrame
    pair_summary: pd.DataFrame
    binned_class_rates: dict[str, pd.DataFrame]
    monthly_class_rates: pd.DataFrame
    class_autocorrelations: pd.DataFrame
    leakage_summary: pd.DataFrame
    train_rows: int
    test_rows: int


def _quantile_edges(values: pd.Series, bin_count: int) -> np.ndarray:
    """Fit distinct quantile-bin edges on an earlier partition only."""
    edges = np.unique(np.quantile(values.to_numpy(), np.linspace(0, 1, bin_count + 1)))
    if len(edges) < 3:
        raise ValueError("A feature needs at least two distinct values for quantile binning.")
    return edges


def _assign_quantile_bins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Apply previously fitted quantile edges without learning from later data."""
    return np.searchsorted(edges[1:-1], values.to_numpy(), side="right")


def _eta_squared(values: pd.Series, target: pd.Series, classes: tuple[int, ...]) -> float:
    """Return the fraction of a feature's variance associated with class membership."""
    overall_mean = values.mean()
    total_sum_squares = float(((values - overall_mean) ** 2).sum())
    if total_sum_squares == 0:
        return 0.0

    between_sum_squares = sum(
        int((target == class_value).sum())
        * float((values.loc[target == class_value].mean() - overall_mean) ** 2)
        for class_value in classes
    )
    return float(between_sum_squares / total_sum_squares)


def _one_vs_rest_auc(values: pd.Series, target: pd.Series, classes: tuple[int, ...]) -> tuple[float, float, int]:
    """Return macro and maximum orientation-neutral one-vs-rest AUCs."""
    class_aucs: list[tuple[int, float]] = []
    for class_value in classes:
        is_class = (target == class_value).astype(int)
        raw_auc = roc_auc_score(is_class, values)
        class_aucs.append((class_value, float(max(raw_auc, 1 - raw_auc))))

    best_class, best_auc = max(class_aucs, key=lambda item: item[1])
    return (
        float(np.mean([auc for _, auc in class_aucs])),
        best_auc,
        best_class,
    )


def _class_rates_by_bin(
    train_values: pd.Series,
    train_target: pd.Series,
    test_values: pd.Series,
    test_target: pd.Series,
    classes: tuple[int, ...],
    bin_count: int = 10,
) -> pd.DataFrame:
    """Return per-class bin rates using quantile edges learned from training only."""
    edges = _quantile_edges(train_values, bin_count)
    train_bins = _assign_quantile_bins(train_values, edges)
    test_bins = _assign_quantile_bins(test_values, edges)
    actual_bin_count = len(edges) - 1

    result = pd.DataFrame({"bin": range(actual_bin_count)})
    for partition, bins, target in (
        ("train", train_bins, train_target),
        ("test", test_bins, test_target),
    ):
        crosstab = pd.crosstab(bins, target).reindex(
            index=range(actual_bin_count), columns=classes, fill_value=0
        )
        counts = crosstab.sum(axis=1)
        result[f"{partition}_count"] = counts.to_numpy()
        for class_value in classes:
            result[f"{partition}_rate_{class_value}"] = np.divide(
                crosstab[class_value].to_numpy(),
                counts.to_numpy(),
                out=np.zeros(actual_bin_count, dtype=float),
                where=counts.to_numpy() > 0,
            )
    return result


def _cell_class_probabilities(
    bins: np.ndarray, target: np.ndarray, classes: tuple[int, ...], bin_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-grid-cell observation counts and conditional class probabilities."""
    cell_index = bins[:, 0] * bin_count + bins[:, 1]
    counts = np.bincount(cell_index, minlength=bin_count**2)
    class_counts = np.column_stack(
        [
            np.bincount(cell_index, weights=(target == class_value), minlength=bin_count**2)
            for class_value in classes
        ]
    )
    probabilities = np.divide(
        class_counts,
        counts[:, None],
        out=np.zeros_like(class_counts, dtype=float),
        where=counts[:, None] > 0,
    )
    return counts, probabilities


def _pairwise_class_pattern(
    train_bins: np.ndarray,
    train_target: np.ndarray,
    test_bins: np.ndarray,
    test_target: np.ndarray,
    classes: tuple[int, ...],
    bin_count: int,
) -> tuple[float, float, float]:
    """Measure class-distribution deviations and their train/test agreement."""
    train_counts, train_probabilities = _cell_class_probabilities(
        train_bins, train_target, classes, bin_count
    )
    test_counts, test_probabilities = _cell_class_probabilities(
        test_bins, test_target, classes, bin_count
    )
    train_prior = np.array([(train_target == class_value).mean() for class_value in classes])
    test_prior = np.array([(test_target == class_value).mean() for class_value in classes])
    train_tv = 0.5 * np.abs(train_probabilities - train_prior).sum(axis=1)
    test_tv = 0.5 * np.abs(test_probabilities - test_prior).sum(axis=1)
    train_valid = train_tv[train_counts >= 100]
    test_valid = test_tv[test_counts >= 100]
    train_max = float(train_valid.max()) if len(train_valid) else 0.0
    test_max = float(test_valid.max()) if len(test_valid) else 0.0

    common_cells = (train_counts >= 100) & (test_counts >= 100)
    train_residual = (train_probabilities[common_cells] - train_prior).ravel()
    test_residual = (test_probabilities[common_cells] - test_prior).ravel()
    if (
        common_cells.sum() < 2
        or np.std(train_residual) == 0
        or np.std(test_residual) == 0
    ):
        agreement = 0.0
    else:
        agreement = float(np.corrcoef(train_residual, test_residual)[0, 1])
    return train_max, test_max, agreement


def _class_distribution(
    train_target: pd.Series, test_target: pd.Series, classes: tuple[int, ...]
) -> pd.DataFrame:
    """Return count and proportion for every documented class in both partitions."""
    rows: list[dict[str, int | float | str]] = []
    for partition, target in (("train", train_target), ("test", test_target)):
        counts = target.value_counts().reindex(classes, fill_value=0)
        for class_value in classes:
            rows.append(
                {
                    "partition": partition,
                    "class": class_value,
                    "class_name": MODEL_2_CLASS_LABELS[class_value],
                    "count": int(counts[class_value]),
                    "proportion": float(counts[class_value] / len(target)),
                }
            )
    return pd.DataFrame(rows)


def _leakage_summary(
    train_data: pd.DataFrame, train_features: pd.DataFrame, train_target: pd.Series
) -> pd.DataFrame:
    """Document forbidden columns and their descriptive relationship to Model 2 classes."""
    rows: list[dict[str, object]] = []
    for column in MODEL_2_EXCLUDED_COLUMNS:
        row: dict[str, object] = {
            "column": column,
            "present_in_X": column in train_features.columns,
            "dtype": str(train_data[column].dtype),
        }
        if column in {"Failure_Probability", "RUL", "TTF", "Component_Health_Score"}:
            row["eta_squared_with_target"] = _eta_squared(
                train_data[column], train_target, MAINTENANCE_CLASSES
            )
            class_means = train_data.groupby(MODEL_2_TARGET)[column].mean()
            row["class_mean_min"] = float(class_means.min())
            row["class_mean_max"] = float(class_means.max())
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_maintenance_type_signal(dataframe: pd.DataFrame) -> MaintenanceTypeSignalAnalysis:
    """Describe Model 2 signal without fitting a multiclass classifier.

    The earliest chronological 80% supplies all feature summaries and bin
    definitions. The later 20% is used only to inspect whether bin and pair
    patterns persist. No learned preprocessing or classification occurs.
    """
    data = parse_timestamp_column(dataframe)
    ensure_target_classes(data, MODEL_2_TARGET, set(MAINTENANCE_CLASSES))
    data = data.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    ensure_valid_timestamps(data)

    train_data, test_data = chronological_train_test_split(data)
    x_train, y_train = get_model_2_xy(train_data)
    x_test, y_test = get_model_2_xy(test_data)

    mutual_information = mutual_info_classif(
        x_train, y_train, discrete_features=False, random_state=42
    )
    feature_rows: list[dict[str, float | int | str]] = []
    class_feature_rows: list[dict[str, float | int | str]] = []
    binned_class_rates: dict[str, pd.DataFrame] = {}
    five_bin_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES):
        values = x_train[feature]
        macro_auc, max_auc, best_class = _one_vs_rest_auc(
            values, y_train, MAINTENANCE_CLASSES
        )
        feature_rows.append(
            {
                "feature": feature,
                "eta_squared": _eta_squared(values, y_train, MAINTENANCE_CLASSES),
                "macro_one_vs_rest_auc": macro_auc,
                "max_one_vs_rest_auc": max_auc,
                "best_separated_class": best_class,
                "best_separated_class_name": MODEL_2_CLASS_LABELS[best_class],
                "mutual_information": float(mutual_information[index]),
            }
        )
        binned_class_rates[feature] = _class_rates_by_bin(
            values, y_train, x_test[feature], y_test, MAINTENANCE_CLASSES
        )
        five_edges = _quantile_edges(values, bin_count=5)
        five_bin_values[feature] = (
            _assign_quantile_bins(values, five_edges),
            _assign_quantile_bins(x_test[feature], five_edges),
        )

        for class_value in MAINTENANCE_CLASSES:
            class_values = values.loc[y_train == class_value]
            class_feature_rows.append(
                {
                    "feature": feature,
                    "class": class_value,
                    "class_name": MODEL_2_CLASS_LABELS[class_value],
                    "count": int(len(class_values)),
                    "mean": float(class_values.mean()),
                    "median": float(class_values.median()),
                    "std": float(class_values.std(ddof=1)),
                    "q05": float(class_values.quantile(QUANTILES[0])),
                    "q25": float(class_values.quantile(QUANTILES[1])),
                    "q75": float(class_values.quantile(QUANTILES[3])),
                    "q95": float(class_values.quantile(QUANTILES[-1])),
                }
            )

    feature_summary = pd.DataFrame(feature_rows).sort_values(
        ["mutual_information", "eta_squared"], ascending=False, kind="stable"
    ).reset_index(drop=True)
    class_feature_summary = pd.DataFrame(class_feature_rows)

    pair_rows: list[dict[str, float | str]] = []
    for first_index, first_feature in enumerate(MODEL_1_OPERATIONAL_FEATURES):
        for second_feature in MODEL_1_OPERATIONAL_FEATURES[first_index + 1 :]:
            train_pair_bins = np.column_stack(
                (five_bin_values[first_feature][0], five_bin_values[second_feature][0])
            )
            test_pair_bins = np.column_stack(
                (five_bin_values[first_feature][1], five_bin_values[second_feature][1])
            )
            train_max, test_max, agreement = _pairwise_class_pattern(
                train_pair_bins,
                y_train.to_numpy(),
                test_pair_bins,
                y_test.to_numpy(),
                MAINTENANCE_CLASSES,
                bin_count=5,
            )
            pair_rows.append(
                {
                    "feature_a": first_feature,
                    "feature_b": second_feature,
                    "train_max_class_distribution_deviation": train_max,
                    "test_max_class_distribution_deviation": test_max,
                    "train_test_pattern_correlation": agreement,
                }
            )
    pair_summary = pd.DataFrame(pair_rows).sort_values(
        "train_max_class_distribution_deviation", ascending=False, kind="stable"
    ).reset_index(drop=True)

    timestamp_indexed = data.set_index("Timestamp")
    monthly_counts = timestamp_indexed[MODEL_2_TARGET].resample("ME").count()
    complete_months = monthly_counts >= 28 * 24 * 4
    monthly_class_rates = pd.crosstab(
        timestamp_indexed.index.to_period("M"), timestamp_indexed[MODEL_2_TARGET], normalize="index"
    ).reindex(columns=MAINTENANCE_CLASSES, fill_value=0)
    monthly_class_rates.index = monthly_class_rates.index.to_timestamp(how="end").normalize()
    monthly_class_rates = monthly_class_rates.loc[complete_months]
    monthly_class_rates.columns = [
        f"{class_value}_{MODEL_2_CLASS_LABELS[class_value]}" for class_value in MAINTENANCE_CLASSES
    ]

    autocorrelation_rows = []
    for class_value in MAINTENANCE_CLASSES:
        class_target = (y_train == class_value).astype(int)
        for lag in (1, 4, 96, 672):
            autocorrelation_rows.append(
                {
                    "class": class_value,
                    "class_name": MODEL_2_CLASS_LABELS[class_value],
                    "lag": lag,
                    "autocorrelation": float(class_target.autocorr(lag)),
                }
            )

    return MaintenanceTypeSignalAnalysis(
        class_distribution=_class_distribution(y_train, y_test, MAINTENANCE_CLASSES),
        feature_summary=feature_summary,
        class_feature_summary=class_feature_summary,
        pair_summary=pair_summary,
        binned_class_rates=binned_class_rates,
        monthly_class_rates=monthly_class_rates,
        class_autocorrelations=pd.DataFrame(autocorrelation_rows),
        leakage_summary=_leakage_summary(train_data, x_train, y_train),
        train_rows=len(train_data),
        test_rows=len(test_data),
    )
