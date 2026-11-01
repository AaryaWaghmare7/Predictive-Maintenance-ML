"""Descriptive dashboard data, never model fitting, scaling or safety thresholds."""

from functools import lru_cache
from itertools import combinations
import json

import numpy as np
import pandas as pd

from src.config.paths import METRICS_DIR
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, NORMALIZED_RANGE_TOLERANCE

REFERENCE_PATH = METRICS_DIR / "experiment_2" / "dashboard_reference.json"
HISTOGRAM_BINS = 35
SCATTER_LIMIT = 1000
SHAPE_REVIEW_DISTANCE = 0.25  # Descriptive review heuristic, not a validated OOD test.
SHAPE_REVIEW_MIN_ROWS = 30


@lru_cache(maxsize=1)
def load_dashboard_reference():
    """Small aggregate snapshot; raw data and fitted estimators are not served."""
    if not REFERENCE_PATH.is_file():
        return None
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _feature_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {"count": int(len(finite)), "excluded_nonfinite": int(len(values)-len(finite)),
              "min": None, "q1": None, "median": None, "q3": None, "max": None,
              "mean": None, "std": None, "histogram": [0]*HISTOGRAM_BINS,
              "below_zero": 0, "above_one": 0, "actual_range_histogram": None}
    if not len(finite):
        return result
    for key, value in zip(("min", "q1", "median", "q3", "max"), np.quantile(finite, [0, .25, .5, .75, 1])):
        result[key] = float(value) if np.isfinite(value) else None
    mean, std = finite.mean(), finite.std(ddof=1) if len(finite) > 1 else np.nan
    result.update(mean=float(mean) if np.isfinite(mean) else None,
                  std=float(std) if np.isfinite(std) else None)
    tol = NORMALIZED_RANGE_TOLERANCE
    inside = finite[(finite >= -tol) & (finite <= 1+tol)]
    # Bin assignment tolerates documented floating-point roundoff; values themselves
    # are never clipped, converted or used to fit a normalization transform.
    bins = np.minimum(HISTOGRAM_BINS-1, np.maximum(0, np.floor(inside*HISTOGRAM_BINS).astype(int)))
    result["histogram"] = np.bincount(bins, minlength=HISTOGRAM_BINS).tolist()
    result["below_zero"] = int((finite < -tol).sum())
    result["above_one"] = int((finite > 1+tol).sum())
    if (result["below_zero"] or result["above_one"]) and np.isfinite(finite.max()-finite.min()):
        if finite.min() == finite.max():
            counts, edges = [len(finite)], [float(finite[0]), float(finite[0])]
        else:
            counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS)
            counts, edges = counts.tolist(), edges.tolist()
        result["actual_range_histogram"] = {"counts": counts, "edges": edges}
    return result


def _group_summary(sensors):
    correlation = sensors.corr(min_periods=3)
    return {"rows": len(sensors),
        "features": {feature: _feature_summary(sensors[feature]) for feature in NEV_FEATURES},
        "correlation": [[float(value) if np.isfinite(value) else None for value in row]
                        for row in correlation.to_numpy()],
        "pair_counts": (sensors.notna().astype(int).T @ sensors.notna().astype(int)).to_numpy().tolist()}


def _scatter_sample(sensors, labels):
    """Bounded, deterministic display sample only; never changes the ML split."""
    indices = []
    codes = sorted(labels.unique())
    budget = max(1, SCATTER_LIMIT // len(codes)) if codes else 0
    for code in codes:
        positions = np.flatnonzero(labels.to_numpy() == code)
        if len(positions) > budget:
            positions = np.random.default_rng(42).choice(positions, budget, replace=False)
        indices.extend(positions.tolist())
    return [{"row": index+1, "class": str(labels.iloc[index]),
             "values": [float(value) if np.isfinite(value) else None for value in sensors.iloc[index]]}
            for index in sorted(indices)]


def binned_reference_scatter(sensors, labels, bins=20):
    """Joint-bin counts for original data; never store raw training observations."""
    pairs = {}
    for first, second in combinations(range(len(NEV_FEATURES)), 2):
        groups = {}
        for code in NEV_CLASSES:
            values = sensors.loc[labels == code, [NEV_FEATURES[first], NEV_FEATURES[second]]].to_numpy()
            cells = np.clip(np.floor(values*bins).astype(int), 0, bins-1)
            counts = np.bincount(cells[:, 0]*bins+cells[:, 1], minlength=bins*bins)
            groups[str(code)] = [[int(i//bins), int(i % bins), int(count)] for i, count in enumerate(counts) if count]
        pairs[f"{first}:{second}"] = groups
    return {"kind": "binned_reference", "bins": bins, "pairs": pairs,
            "note": "20×20 joint-bin counts, not exact training rows. Overlapping classes may share bins."}


def training_comparison(sensors, reference, normalized):
    if reference is None:
        return {"available": False, "reason": "Verified training aggregate snapshot is unavailable"}
    train = reference["datasets"]["training"]["groups"]["all"]["features"]
    comparisons = {}
    for feature in NEV_FEATURES:
        values = sensors[feature].dropna().to_numpy()
        observed, base = _feature_summary(values), train[feature]
        tol = NORMALIZED_RANGE_TOLERANCE
        outside = int(((values < base["min"]-tol) | (values > base["max"]+tol)).sum())
        obs_bins = np.asarray([observed["below_zero"], *observed["histogram"], observed["above_one"]])
        train_bins = np.asarray([base["below_zero"], *base["histogram"], base["above_one"]])
        distance = float(np.abs(obs_bins/len(values)-train_bins/base["count"]).sum()/2) if len(values) else None
        comparisons[feature] = {"training_min": base["min"], "training_max": base["max"],
            "training_median": base["median"], "observed_min": observed["min"], "observed_max": observed["max"],
            "observed_median": observed["median"], "outside_training_range": outside,
            "finite_rows": len(values), "histogram_total_variation": distance,
            "shape_review": bool(len(values) >= SHAPE_REVIEW_MIN_ROWS and distance > SHAPE_REVIEW_DISTANCE)}
    return {"available": True, "normalization_confirmed": bool(normalized), "features": comparisons,
        "method": "Training-only observed min/max (1e-12 tolerance) and total variation between 35 fixed bins plus under/overflow. Shape review: distance >0.25 with at least 30 finite observations; a configurable descriptive heuristic, not statistical significance or a validated OOD detector.",
        "limitation": "Numeric similarity cannot establish compatible units, vehicle context or real-world validity. Differences may reflect class mix or normalization. No engineering safety limit, transformation or automatic row deletion is implied."}


def build_dashboard_data(frame, *, labels=None, label_basis="unlabeled", normalized=False,
                         reference=None, include_scatter=True):
    """Compute statistics from all upload rows, not the capped prediction list.

    Only chart calculations exclude nonnumeric/nonfinite cells, with exact
    counts. Pairwise correlations report sample counts and null for undefined
    values. These exclusions never make invalid inputs eligible for prediction.
    """
    sensors = frame.reindex(columns=NEV_FEATURES).apply(pd.to_numeric, errors="coerce")
    sensors = sensors.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if labels is None:
        codes = pd.Series([-1]*len(sensors), dtype=int)
    else:
        codes = pd.Series(np.asarray(labels))
        if len(codes) != len(sensors) or not codes.isin(NEV_CLASSES).all():
            raise ValueError("Dashboard label alignment must match all four supported categories")
        codes = codes.astype(int)
    groups = {"all": _group_summary(sensors)}
    if labels is not None:
        groups.update({str(code): _group_summary(sensors.loc[codes == code]) for code in NEV_CLASSES})
    associations = {}
    if labels is not None:
        for feature in NEV_FEATURES:
            usable = sensors[feature].notna()
            values, target = sensors.loc[usable, feature], codes[usable]
            total = float(((values-values.mean())**2).sum())
            if total > 0 and target.nunique() >= 2:
                between = sum(int((target == code).sum()) * (float(values[target == code].mean())-float(values.mean()))**2
                              for code in target.unique())
                associations[feature] = {"eta_squared": min(1., max(0., between/total))
                    if np.isfinite(total) and np.isfinite(between) else None}
            else:
                associations[feature] = {"eta_squared": None}
    quality = {feature: {"missing_or_invalid_for_charts": groups["all"]["features"][feature]["excluded_nonfinite"],
                        "outside_normalized_range": groups["all"]["features"][feature]["below_zero"]+groups["all"]["features"][feature]["above_one"]}
               for feature in NEV_FEATURES}
    return {"version": "dashboard-1", "rows": len(sensors), "features": list(NEV_FEATURES),
        "class_names": {str(code): name for code, name in NEV_CLASSES.items()}, "label_basis": label_basis,
        "class_counts": {str(code): int((codes == code).sum()) for code in NEV_CLASSES} if labels is not None else None,
        "normalized": bool(normalized), "groups": groups, "quality": quality,
        "associations": associations, "scatter": {"kind": "sample", "points": _scatter_sample(sensors, codes),
            "limit": SCATTER_LIMIT, "note": "Deterministic class-stratified display sample, at most 1,000 rows. Aggregates use every observation; sampling does not change model inputs or the train/test split."} if include_scatter else None,
        "comparison": training_comparison(sensors, reference, normalized),
        "caution": "Descriptive associations, correlation and model importance do not imply causation. Labels from model predictions are not ground truth. No temporal, cross-vehicle or OEM validation is established."}
