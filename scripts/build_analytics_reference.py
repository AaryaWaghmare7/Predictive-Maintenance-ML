"""Build aggregate chart data from hash-verified NEV files; never train models.

Run explicitly with the original local dataset. No raw records are exported.
Historical MI/importance are copied from saved analysis, not recomputed/fitted.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import METRICS_DIR
from src.data.nev_fault import load_nev_datasets, NEV_DATA_DIR, NEV_FILES, file_sha256, NEV_FEATURES, NEV_TARGET
from src.eda.dashboard import REFERENCE_PATH, build_dashboard_data, binned_reference_scatter


def main():
    metadata = json.loads((PROJECT_ROOT / "models/experiment_2/metadata.json").read_text())
    hashes = {name: file_sha256(NEV_DATA_DIR / filename) for name, filename in NEV_FILES.items()}
    if hashes != metadata["raw_sha256"]:
        raise ValueError("Raw files do not match the recorded Experiment 2 fingerprints")
    frames = load_nev_datasets()
    train = frames["training"]
    training = build_dashboard_data(train, labels=train[NEV_TARGET], label_basis="original_training_labels",
                                    normalized=True, include_scatter=False)
    training["scatter"] = binned_reference_scatter(train[list(NEV_FEATURES)], train[NEV_TARGET])
    full = build_dashboard_data(frames["full"], normalized=True, include_scatter=False,
                               label_basis="original_full_descriptive_only")
    full["class_counts"] = {str(c): int((frames["full"][NEV_TARGET] == c).sum()) for c in range(4)}
    for data in (training, full):
        data["comparison"] = {"available": False,
            "reason": "Original reference data is not an uploaded vehicle. Select an uploaded source to compare against training."}
    signal = json.loads((METRICS_DIR / "experiment_2/signal.json").read_text())
    models = json.loads((METRICS_DIR / "experiment_2/model_comparison.json").read_text())
    reference = {"version": "nev-reference-1", "source_report": "reports/09_NEV_EDA_Report.md",
        "raw_sha256": hashes, "datasets": {"training": training, "full": full},
        "historical_associations": signal["associations"],
        "historical_importance": models["feature_importance"],
        "final_model_impurity_importance": models["final_training_impurity_importance"],
        "scope": "Full 11,000: descriptive histograms/correlation. Training 7,700: class boxplots, joint-bin scatter and associations. Saved training-validation permutation importance and final-model impurity importance are historical, not upload-specific or causal."}
    REFERENCE_PATH.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n", encoding="utf-8")
    print(f"Created {REFERENCE_PATH.relative_to(PROJECT_ROOT)} ({REFERENCE_PATH.stat().st_size:,} bytes); aggregate data only; no fitting")


if __name__ == "__main__":
    main()
