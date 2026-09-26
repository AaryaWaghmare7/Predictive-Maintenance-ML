"""Run the Model 1 failure-signal investigation without fitting a classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import FIGURES_DIR, RAW_DATA_DIR
from src.data.load_data import load_dataset
from src.eda.failure_signal import analyze_failure_signal


def save_failure_signal_plots(analysis, output_dir: Path) -> list[Path]:
    """Save comparison, binned-rate, and temporal diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    top_features = analysis.feature_summary.head(4)["feature"].tolist()
    saved_paths: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, feature in zip(axes.flat, top_features):
        summary = analysis.feature_summary.set_index("feature").loc[feature]
        axis.bar(
            ["No Failure", "Failure"],
            [summary["no_failure_median"], summary["failure_median"]],
            color=["#4C78A8", "#F58518"],
        )
        axis.set_title(feature)
        axis.set_ylabel("Median feature value")
    figure.suptitle("Top individual associations: class medians")
    path = output_dir / "failure_signal_group_medians.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, feature in zip(axes.flat, top_features):
        rates = analysis.binned_failure_rates[feature]
        axis.plot(rates["bin"] + 1, rates["train_failure_rate"], marker="o", label="Train")
        axis.plot(rates["bin"] + 1, rates["test_failure_rate"], marker="s", label="Test")
        axis.axhline(analysis.train_failure_rate, color="gray", linestyle="--", label="Train average")
        axis.set_title(feature)
        axis.set_xlabel("Training-defined quantile bin")
        axis.set_ylabel("Failure rate")
        axis.set_ylim(0, max(0.2, axis.get_ylim()[1]))
        axis.legend(fontsize=8)
    figure.suptitle("Failure rate by training-defined feature bins")
    path = output_dir / "failure_signal_binned_rates.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(analysis.monthly_target_rate.index, analysis.monthly_target_rate.values, color="#E45756")
    axes[0].set_title("Monthly failure rate")
    axes[0].set_ylabel("Failure rate")
    for axis, feature in zip(axes[1:], top_features[:2]):
        axis.plot(
            analysis.monthly_feature_means.index,
            analysis.monthly_feature_means[feature],
            color="#4C78A8",
        )
        axis.set_title(f"Monthly mean: {feature}")
        axis.set_ylabel("Feature value")
    axes[-1].set_xlabel("Month")
    path = output_dir / "failure_signal_temporal_patterns.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    return saved_paths


def main() -> None:
    """Run descriptive analysis and save figures without training a model."""
    dataset_path = RAW_DATA_DIR / "EV_Predictive_Maintenance_Dataset_15min.csv"
    analysis = analyze_failure_signal(load_dataset(dataset_path))
    plot_paths = save_failure_signal_plots(analysis, FIGURES_DIR)
    print(analysis.feature_summary.head(10).to_string(index=False))
    print("\nTop pairwise bin deviations:")
    print(analysis.pair_summary.head(10).to_string(index=False))
    print("\nSaved plots:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
