"""Run the Model 2 maintenance-type signal investigation without training a model."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import FIGURES_DIR, RAW_DATA_DIR
from src.data.load_data import load_dataset
from src.eda.maintenance_type_signal import analyze_maintenance_type_signal
from src.preprocessing.cleaning import MODEL_2_CLASS_LABELS


CLASS_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#B279A2",
}


def save_maintenance_type_signal_plots(analysis, output_dir: Path) -> list[Path]:
    """Save class, feature, nonlinear-bin, and temporal diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_values = list(MODEL_2_CLASS_LABELS)
    class_names = [MODEL_2_CLASS_LABELS[class_value] for class_value in class_values]
    top_features = analysis.feature_summary.head(4)["feature"].tolist()
    saved_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    distribution = analysis.class_distribution
    positions = np.arange(len(class_values))
    width = 0.35
    for offset, partition in ((-width / 2, "train"), (width / 2, "test")):
        counts = (
            distribution.loc[distribution["partition"] == partition]
            .set_index("class")
            .loc[class_values, "count"]
        )
        axis.bar(positions + offset, counts, width, label=partition.title())
    axis.set_xticks(positions, class_names)
    axis.set_ylabel("Rows")
    axis.set_title("Maintenance_Type class distribution")
    axis.legend()
    path = output_dir / "maintenance_type_class_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    class_summary = analysis.class_feature_summary
    for axis, feature in zip(axes.flat, top_features):
        medians = (
            class_summary.loc[class_summary["feature"] == feature]
            .set_index("class")
            .loc[class_values, "median"]
        )
        axis.bar(
            class_names,
            medians,
            color=[CLASS_COLORS[class_value] for class_value in class_values],
        )
        axis.set_title(feature)
        axis.set_ylabel("Class median")
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Class medians for the four strongest association screens")
    path = output_dir / "maintenance_type_class_medians.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    train_rates = (
        distribution.loc[distribution["partition"] == "train"]
        .set_index("class")["proportion"]
        .to_dict()
    )
    test_rates = (
        distribution.loc[distribution["partition"] == "test"]
        .set_index("class")["proportion"]
        .to_dict()
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, feature in zip(axes.flat, top_features):
        rates = analysis.binned_class_rates[feature]
        for class_value in class_values:
            axis.plot(
                rates["bin"] + 1,
                rates[f"train_rate_{class_value}"] / train_rates[class_value],
                color=CLASS_COLORS[class_value],
                marker="o",
                label=f"{MODEL_2_CLASS_LABELS[class_value]} train",
            )
            axis.plot(
                rates["bin"] + 1,
                rates[f"test_rate_{class_value}"] / test_rates[class_value],
                color=CLASS_COLORS[class_value],
                marker="s",
                linestyle="--",
                label=f"{MODEL_2_CLASS_LABELS[class_value]} test",
            )
        axis.axhline(1.0, color="gray", linestyle=":")
        axis.set_title(feature)
        axis.set_xlabel("Training-defined quantile bin")
        axis.set_ylabel("Class rate / partition rate")
        axis.legend(fontsize=6, ncol=2)
    figure.suptitle("Class-rate ratios by feature bin: train versus later test")
    path = output_dir / "maintenance_type_binned_class_rates.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for class_value in class_values:
        column = f"{class_value}_{MODEL_2_CLASS_LABELS[class_value]}"
        axis.plot(
            analysis.monthly_class_rates.index,
            analysis.monthly_class_rates[column],
            color=CLASS_COLORS[class_value],
            label=f"{class_value}: {MODEL_2_CLASS_LABELS[class_value]}",
        )
    axis.set_title("Monthly Maintenance_Type class proportions")
    axis.set_xlabel("Month")
    axis.set_ylabel("Class proportion")
    axis.legend(ncol=2)
    path = output_dir / "maintenance_type_temporal_patterns.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    saved_paths.append(path)

    return saved_paths


def main() -> None:
    """Run and print descriptive Model 2 results without fitting a classifier."""
    dataset_path = RAW_DATA_DIR / "EV_Predictive_Maintenance_Dataset_15min.csv"
    analysis = analyze_maintenance_type_signal(load_dataset(dataset_path))
    plot_paths = save_maintenance_type_signal_plots(analysis, FIGURES_DIR)
    print("Class distribution:")
    print(analysis.class_distribution.to_string(index=False))
    print("\nTop feature association screens:")
    print(analysis.feature_summary.head(10).to_string(index=False))
    print("\nTop pairwise class-distribution deviations:")
    print(analysis.pair_summary.head(10).to_string(index=False))
    print("\nLeakage-safe input checks:")
    print(analysis.leakage_summary.to_string(index=False))
    print("\nSaved plots:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
