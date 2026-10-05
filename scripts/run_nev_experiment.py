"""Approved Experiment 2 audit/EDA/CV/final evaluation, with no Git actions.

Run from any working directory: python /path/to/project/scripts/run_nev_experiment.py
Raw input defaults are project-relative. Generated artifacts remain local.
Do not repeatedly rerun this script to improve already-observed test scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config.paths import FIGURES_DIR, METRICS_DIR, MODEL_DIR
from src.data.nev_fault import NEV_CLASSES, NEV_DATA_DIR, NEV_FEATURES, NEV_FILES, file_sha256, load_nev_datasets
from src.eda.nev_fault import audit_nev_datasets, investigate_nev_signal, save_nev_plots
from src.prediction.nev_fault import predict_nev_faults
from src.training.nev_fault import build_nev_candidates, explain_nev_tree, fit_and_evaluate_nev, select_nev_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=NEV_DATA_DIR)
    args = parser.parse_args()
    fingerprints = {name: file_sha256(args.data_dir / filename) for name, filename in NEV_FILES.items()}
    frames = load_nev_datasets(args.data_dir)
    audit = audit_nev_datasets(frames)
    print("Audit: supplied split verified; raw files unchanged", flush=True)
    signal = investigate_nev_signal(frames["training"])
    print(f"Shallow-rule training CV macro F1: {signal['shallow_rule_cv_macro_f1']['mean']:.6f}", flush=True)
    # Persist pre-test evidence before fitting/evaluating test candidates.
    output_dir = METRICS_DIR / "experiment_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("audit.json", audit), ("signal.json", signal)):
        (output_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    selection = select_nev_model(frames["training"])
    (output_dir / "selection_before_test.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selected BEFORE test evaluation: {selection['selected_model']}", flush=True)
    importance = explain_nev_tree(frames["training"], selection["best_tree"])
    models, test_results = fit_and_evaluate_nev(frames["training"], frames["testing"], selection)
    selected = selection["selected_model"]
    destination = MODEL_DIR / "experiment_2"
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "nev_fault_pipeline.joblib"
    joblib.dump(models[selected], model_path)
    restored = joblib.load(model_path)
    sample = frames["testing"].loc[:, list(NEV_FEATURES)].head(20)
    pd.testing.assert_frame_equal(predict_nev_faults(models[selected], sample), predict_nev_faults(restored, sample))
    assert fingerprints == {name: file_sha256(args.data_dir / filename) for name, filename in NEV_FILES.items()}, "Raw inputs changed"
    metadata = {
        "selected_model": selected, "selection_criterion": "highest mean training-only 5-fold macro F1",
        "feature_order": list(NEV_FEATURES), "target_mapping": NEV_CLASSES,
        "input_scale": "normalized [0,1]; original normalization formula unknown",
        "split": "supplied 7700 training / 3300 testing; no new train_test_split",
        "raw_sha256": fingerprints, "selection": selection, "test_results": test_results,
        "feature_importance": importance,
        "final_training_impurity_importance": dict(zip(NEV_FEATURES, models[selection["best_tree"]].named_steps["classifier"].feature_importances_.tolist()))
        if hasattr(models[selection["best_tree"]].named_steps["classifier"], "feature_importances_") else None,
        "configurations": {name: {key: value for key, value in pipeline.named_steps["classifier"].get_params().items()} for name, pipeline in build_nev_candidates().items()},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "optional_dependencies": {name: importlib.util.find_spec(name) is not None for name in ("xgboost", "shap")},
        "inference_round_trip_verified": True, "raw_hashes_unchanged": True,
    }
    payload = json.dumps(metadata, indent=2, ensure_ascii=False)
    (destination / "metadata.json").write_text(payload, encoding="utf-8")
    (output_dir / "model_comparison.json").write_text(payload, encoding="utf-8")
    save_nev_plots(frames, signal, FIGURES_DIR / "experiment_2", test_results, importance)
    for name, result in test_results.items():
        print(name, {key: value for key, value in result.items() if key not in ("per_class",)}, flush=True)
    print("Saved pipeline and metadata under models/experiment_2/; no Git actions performed", flush=True)


if __name__ == "__main__":
    main()
