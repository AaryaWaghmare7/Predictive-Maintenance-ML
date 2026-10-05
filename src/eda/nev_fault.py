"""Reproducible Experiment 2 audit, training-only signal checks and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, NEV_TARGET, NORMALIZED_RANGE_TOLERANCE, get_nev_xy, verify_supplied_split


def audit_nev_datasets(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Measure issues and IQR flags; never remove rows or treat flags as errors.

    Train-derived IQR fences are used on all partitions for comparison only.
    Descriptive full/test summaries never feed preprocessing or model choice.
    """
    compatibility = verify_supplied_split(frames["full"], frames["training"], frames["testing"])
    train = frames["training"].loc[:, list(NEV_FEATURES)]
    q1, q3 = train.quantile(0.25), train.quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    datasets = {}
    for name, frame in frames.items():
        sensors = frame.loc[:, list(NEV_FEATURES)]
        datasets[name] = {
            "shape": list(frame.shape), "columns": frame.columns.tolist(),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": frame.isna().sum().astype(int).to_dict(),
            "infinite_values": int(np.isinf(frame.to_numpy()).sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_feature_rows": int(sensors.duplicated().sum()),
            "class_counts": {str(int(code)): int(count) for code, count in frame[NEV_TARGET].value_counts().sort_index().items()},
            "class_proportions": {str(int(code)): float(count) for code, count in frame[NEV_TARGET].value_counts(normalize=True).sort_index().items()},
            "summary": sensors.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
            "outside_unit_interval": ((sensors < 0) | (sensors > 1)).sum().astype(int).to_dict(),
            "outside_unit_interval_beyond_tolerance": ((sensors < -NORMALIZED_RANGE_TOLERANCE) | (sensors > 1 + NORMALIZED_RANGE_TOLERANCE)).sum().astype(int).to_dict(),
            "iqr_flags_using_training_fences": ((sensors < lower) | (sensors > upper)).sum().astype(int).to_dict(),
        }
    return {"datasets": datasets, "split": compatibility,
            "iqr_training_fences": {feature: {"lower": float(lower[feature]), "upper": float(upper[feature])} for feature in NEV_FEATURES}}


def investigate_nev_signal(training: pd.DataFrame) -> dict[str, object]:
    """Use training labels only for associations and shallow-rule diagnostics.

    Eta squared measures between-class mean variation; mutual information can
    capture nonlinear association. Numeric label codes have no ordinal meaning,
    so feature-vs-label Pearson correlations would be misleading and are omitted.
    """
    x, y = get_nev_xy(training)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mi = mutual_info_classif(x, y, discrete_features=False, random_state=42)
    associations = {}
    groups = {}
    for index, feature in enumerate(NEV_FEATURES):
        values = x[feature]
        total = float(((values - values.mean()) ** 2).sum())
        between = sum(int((y == code).sum()) * (float(values[y == code].mean()) - float(values.mean())) ** 2 for code in NEV_CLASSES)
        stump = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
        scores = cross_val_score(stump, x[[feature]], y, cv=cv, scoring="f1_macro", error_score="raise")
        associations[feature] = {
            "eta_squared": between / total if total else 0.0,
            "mutual_information_nats": float(mi[index]),
            "univariate_tree_cv_macro_f1": float(scores.mean()),
            "one_vs_rest_auc_direction_free": {str(code): max(float(roc_auc_score(y == code, values)), 1 - float(roc_auc_score(y == code, values))) for code in NEV_CLASSES},
        }
        groups[feature] = {str(code): values[y == code].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict() for code in NEV_CLASSES}
    rule_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    scores = cross_val_score(rule_tree, x, y, cv=cv, scoring="f1_macro", error_score="raise")
    rule_tree.fit(x, y)
    return {
        "scope": "supplied training dataset only", "associations": associations,
        "class_feature_summary": groups,
        "shallow_rule_cv_macro_f1": {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()},
        "shallow_rule_text": export_text(rule_tree, feature_names=list(NEV_FEATURES), decimals=6),
        "sensor_correlation": x.corr().to_dict(),
    }


def save_nev_plots(frames, signal, output_dir: Path, test_results=None, importance=None) -> None:
    """Save labeled static scientific figures; class-dependent plots use train."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    colors = sns.color_palette("colorblind", 4)
    class_names = list(NEV_CLASSES.values())

    def save(figure, filename):
        figure.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)

    fig, ax = plt.subplots(figsize=(9, 4))
    proportions = pd.DataFrame({name: frame[NEV_TARGET].value_counts(normalize=True).sort_index().to_numpy() * 100 for name, frame in frames.items()}, index=class_names)
    proportions.plot.bar(ax=ax, rot=0)
    ax.set(ylabel="Class proportion (%)", title="Supplied full/train/test class proportions")
    save(fig, "class_distribution.png")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for feature, ax in zip(NEV_FEATURES, axes.flat):
        ax.hist(frames["full"][feature], bins=35, color=colors[0], edgecolor="white")
        ax.set(title=feature, xlabel="Normalized value (not physical units)", ylabel="Rows")
    for ax in list(axes.flat)[len(NEV_FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Full-dataset descriptive histograms", y=1.01)
    fig.tight_layout()
    save(fig, "feature_histograms.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(frames["full"].loc[:, list(NEV_FEATURES)].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Full-dataset sensor correlation (descriptive only)")
    save(fig, "sensor_correlation.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for feature, ax in zip(NEV_FEATURES, axes.flat):
        arrays = [frames["training"].loc[frames["training"][NEV_TARGET] == code, feature] for code in NEV_CLASSES]
        boxes = ax.boxplot(arrays, tick_labels=["Normal", "Motor", "Inverter", "Battery"], patch_artist=True, showfliers=False)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
        ax.set(title=feature, ylabel="Normalized value", ylim=(-0.04, 1.04))
        ax.tick_params(axis="x", labelrotation=20)
    for ax in list(axes.flat)[len(NEV_FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Feature distributions by fault class (training only)", y=1.01)
    fig.tight_layout()
    save(fig, "features_by_class.png")

    associations = pd.DataFrame(signal["associations"]).T.sort_values("mutual_information_nats")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, title in zip(axes, ["eta_squared", "mutual_information_nats"], ["Between-class effect (eta squared)", "Nonlinear association (MI, nats)"]):
        associations[column].plot.barh(ax=ax, color=colors[0])
        ax.set_title(title)
    fig.suptitle("Training-only feature-target associations", y=1.05)
    fig.tight_layout()
    save(fig, "feature_associations.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for code, name in NEV_CLASSES.items():
        subset = frames["training"].loc[frames["training"][NEV_TARGET] == code]
        for ax, second in zip(axes, ["Vibration (g)", "Temperature (°C)"]):
            ax.scatter(subset["Voltage (V)"], subset[second], s=5, alpha=0.3, label=name, color=colors[code], rasterized=True)
            ax.set(xlabel="Normalized Voltage", ylabel=f"Normalized {second}")
    axes[1].legend(markerscale=3, loc="upper right")
    fig.suptitle("Sharp joint class boundaries (training only)")
    save(fig, "class_separation.png")

    if test_results:
        fig, axes = plt.subplots(1, len(test_results), figsize=(15, 4))
        for ax, (name, result) in zip(np.atleast_1d(axes), test_results.items()):
            sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cbar=False, cmap="Blues", xticklabels=["N", "M", "I", "B"], yticklabels=["N", "M", "I", "B"], ax=ax)
            ax.set(title=name.replace("_", " "), xlabel="Predicted", ylabel="Actual")
        fig.suptitle("Final supplied test confusion matrices: N/M/I/B", y=1.04)
        fig.tight_layout()
        save(fig, "test_confusion_matrices.png")
    if importance:
        ranks = pd.DataFrame(importance["permutation_macro_f1_drop"]).T.sort_values("mean")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(ranks.index, ranks["mean"], xerr=ranks["std"], color=colors[0])
        ax.set(xlabel="Macro F1 decrease after permutation (mean ± SD)", title=f"{importance['model']}: training-only validation importance")
        save(fig, "feature_importance.png")
