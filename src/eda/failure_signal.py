"""Failure-signal diagnostics for the approved Model 1 feature set.

These helpers describe associations between the fixed operational features and
the binary failure label. They do not fit a classifier, alter the raw data, or
select a future model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from src.preprocessing.cleaning import (
    MODEL_1_OPERATIONAL_FEATURES,
    chronological_train_test_split,
    get_model_1_xy,
    parse_timestamp_column,
)
from src.preprocessing.validation import ensure_binary_target, ensure_valid_timestamps


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class FailureSignalAnalysis:
    """Descriptive results from a chronology-respecting signal investigation."""

    feature_summary: pd.DataFrame
    pair_summary: pd.DataFrame
    binned_failure_rates: dict[str, pd.DataFrame]
    monthly_target_rate: pd.Series
    monthly_feature_means: pd.DataFrame
    target_autocorrelations: dict[int, float]
    train_rows: int
    test_rows: int
    train_failure_rate: float
    test_failure_rate: float


def _cohens_d(no_failure: pd.Series, failure: pd.Series) -> float:
    """Return standardized mean difference using the pooled sample standard deviation."""
    pooled_variance = (
        (len(no_failure) - 1) * no_failure.var(ddof=1)
        + (len(failure) - 1) * failure.var(ddof=1)
    ) / (len(no_failure) + len(failure) - 2)
    if pooled_variance == 0:
        return 0.0
    return float((failure.mean() - no_failure.mean()) / np.sqrt(pooled_variance))


def _quantile_edges(values: pd.Series, bin_count: int) -> np.ndarray:
    """Fit quantile-bin edges on earlier observations only."""
    edges = np.unique(np.quantile(values.to_numpy(), np.linspace(0, 1, bin_count + 1)))
    if len(edges) < 3:
        raise ValueError("A feature needs at least two distinct values for quantile binning.")
    return edges


def _assign_quantile_bins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Apply previously fitted bin edges without learning from later data."""
    return np.searchsorted(edges[1:-1], values.to_numpy(), side="right")


def _binned_failure_rates(
    train_values: pd.Series,
    train_target: pd.Series,
    test_values: pd.Series,
    test_target: pd.Series,
    bin_count: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compare train/test failure rates using bins fitted on the train partition."""
    edges = _quantile_edges(train_values, bin_count)
    train_bins = _assign_quantile_bins(train_values, edges)
    test_bins = _assign_quantile_bins(test_values, edges)
    actual_bin_count = len(edges) - 1

    train_frame = pd.DataFrame({"bin": train_bins, "target": train_target.to_numpy()})
    test_frame = pd.DataFrame({"bin": test_bins, "target": test_target.to_numpy()})
    train_rates = train_frame.groupby("bin")["target"].agg(["count", "mean"])
    test_rates = test_frame.groupby("bin")["target"].agg(["count", "mean"])

    rates = pd.DataFrame({"bin": range(actual_bin_count)})
    rates = rates.merge(
        train_rates.rename(columns={"count": "train_count", "mean": "train_failure_rate"}),
        on="bin",
        how="left",
    ).merge(
        test_rates.rename(columns={"count": "test_count", "mean": "test_failure_rate"}),
        on="bin",
        how="left",
    )
    return rates.fillna(0), edges


def _pairwise_bin_deviation(
    train_bins: np.ndarray,
    train_target: np.ndarray,
    test_bins: np.ndarray,
    test_target: np.ndarray,
    bin_count: int,
) -> tuple[float, float, float]:
    """Return train/test 2D deviations and their cell-rate agreement."""
    def cell_rates(bins: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cell_index = bins[:, 0] * bin_count + bins[:, 1]
        counts = np.bincount(cell_index, minlength=bin_count**2)
        failures = np.bincount(cell_index, weights=target, minlength=bin_count**2)
        rates = np.divide(failures, counts, out=np.zeros_like(failures), where=counts > 0)
        return counts, rates

    train_counts, train_rates = cell_rates(train_bins, train_target)
    test_counts, test_rates = cell_rates(test_bins, test_target)
    train_valid = train_rates[train_counts >= 100]
    test_valid = test_rates[test_counts >= 100]
    train_deviation = (
        float(np.max(np.abs(train_valid - train_target.mean()))) if len(train_valid) else 0.0
    )
    test_deviation = (
        float(np.max(np.abs(test_valid - test_target.mean()))) if len(test_valid) else 0.0
    )
    common_cells = (train_counts >= 100) & (test_counts >= 100)
    if (
        common_cells.sum() < 2
        or np.std(train_rates[common_cells]) == 0
        or np.std(test_rates[common_cells]) == 0
    ):
        agreement = 0.0
    else:
        agreement = float(np.corrcoef(train_rates[common_cells], test_rates[common_cells])[0, 1])
    return train_deviation, test_deviation, agreement


def analyze_failure_signal(dataframe: pd.DataFrame) -> FailureSignalAnalysis:
    """Describe individual, pairwise, and temporal signal without training a model.

    Feature discovery uses the earliest chronological 80%. The latest 20% is
    used only to check whether selected bin-based patterns persist. This is an
    exploratory diagnostic, not a classifier or a model-selection procedure.
    """
    data = parse_timestamp_column(dataframe)
    ensure_binary_target(data, "Failure_Probability")
    data = data.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    ensure_valid_timestamps(data)

    train_data, test_data = chronological_train_test_split(data)
    x_train, y_train = get_model_1_xy(train_data)
    x_test, y_test = get_model_1_xy(test_data)

    mutual_information = mutual_info_classif(
        x_train, y_train, discrete_features=False, random_state=42
    )
    feature_rows: list[dict[str, float | str]] = []
    binned_rates: dict[str, pd.DataFrame] = {}
    five_bin_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for index, feature in enumerate(MODEL_1_OPERATIONAL_FEATURES):
        no_failure = x_train.loc[y_train == 0, feature]
        failure = x_train.loc[y_train == 1, feature]
        binned_rate, _ = _binned_failure_rates(
            x_train[feature], y_train, x_test[feature], y_test
        )
        binned_rates[feature] = binned_rate
        five_edges = _quantile_edges(x_train[feature], bin_count=5)
        five_bin_values[feature] = (
            _assign_quantile_bins(x_train[feature], five_edges),
            _assign_quantile_bins(x_test[feature], five_edges),
        )
        raw_auc = roc_auc_score(y_train, x_train[feature])
        feature_rows.append(
            {
                "feature": feature,
                "no_failure_mean": float(no_failure.mean()),
                "failure_mean": float(failure.mean()),
                "mean_difference": float(failure.mean() - no_failure.mean()),
                "no_failure_median": float(no_failure.median()),
                "failure_median": float(failure.median()),
                "no_failure_std": float(no_failure.std(ddof=1)),
                "failure_std": float(failure.std(ddof=1)),
                "no_failure_q05": float(no_failure.quantile(QUANTILES[0])),
                "failure_q05": float(failure.quantile(QUANTILES[0])),
                "no_failure_q25": float(no_failure.quantile(QUANTILES[1])),
                "failure_q25": float(failure.quantile(QUANTILES[1])),
                "no_failure_q75": float(no_failure.quantile(QUANTILES[3])),
                "failure_q75": float(failure.quantile(QUANTILES[3])),
                "no_failure_q95": float(no_failure.quantile(QUANTILES[-1])),
                "failure_q95": float(failure.quantile(QUANTILES[-1])),
                "cohens_d": _cohens_d(no_failure, failure),
                "point_biserial_correlation": float(x_train[feature].corr(y_train)),
                "univariate_auc": float(max(raw_auc, 1 - raw_auc)),
                "mutual_information": float(mutual_information[index]),
                "train_binned_rate_range": float(
                    binned_rate["train_failure_rate"].max()
                    - binned_rate["train_failure_rate"].min()
                ),
                "test_binned_rate_range": float(
                    binned_rate["test_failure_rate"].max()
                    - binned_rate["test_failure_rate"].min()
                ),
            }
        )

    feature_summary = pd.DataFrame(feature_rows).sort_values(
        "mutual_information", ascending=False, kind="stable"
    ).reset_index(drop=True)

    pair_rows: list[dict[str, float | str]] = []
    for first_index, first_feature in enumerate(MODEL_1_OPERATIONAL_FEATURES):
        for second_feature in MODEL_1_OPERATIONAL_FEATURES[first_index + 1 :]:
            train_pair_bins = np.column_stack(
                (five_bin_values[first_feature][0], five_bin_values[second_feature][0])
            )
            test_pair_bins = np.column_stack(
                (five_bin_values[first_feature][1], five_bin_values[second_feature][1])
            )
            train_deviation, test_deviation, agreement = _pairwise_bin_deviation(
                train_pair_bins,
                y_train.to_numpy(),
                test_pair_bins,
                y_test.to_numpy(),
                bin_count=5,
            )
            pair_rows.append(
                {
                    "feature_a": first_feature,
                    "feature_b": second_feature,
                    "train_max_rate_deviation": train_deviation,
                    "test_max_rate_deviation": test_deviation,
                    "train_test_cell_rate_correlation": agreement,
                }
            )

    pair_summary = pd.DataFrame(pair_rows).sort_values(
        "train_max_rate_deviation", ascending=False, kind="stable"
    ).reset_index(drop=True)
    timestamp_indexed = data.set_index("Timestamp")
    monthly_row_counts = timestamp_indexed["Failure_Probability"].resample("ME").count()
    complete_months = monthly_row_counts >= 28 * 24 * 4
    monthly_target_rate = timestamp_indexed["Failure_Probability"].resample("ME").mean()[
        complete_months
    ]
    monthly_feature_means = (
        timestamp_indexed[list(MODEL_1_OPERATIONAL_FEATURES)]
        .resample("ME")
        .mean()
        .loc[complete_months]
    )

    return FailureSignalAnalysis(
        feature_summary=feature_summary,
        pair_summary=pair_summary,
        binned_failure_rates=binned_rates,
        monthly_target_rate=monthly_target_rate,
        monthly_feature_means=monthly_feature_means,
        target_autocorrelations={
            lag: float(y_train.autocorr(lag)) for lag in (1, 4, 96, 672)
        },
        train_rows=len(train_data),
        test_rows=len(test_data),
        train_failure_rate=float(y_train.mean()),
        test_failure_rate=float(y_test.mean()),
    )
