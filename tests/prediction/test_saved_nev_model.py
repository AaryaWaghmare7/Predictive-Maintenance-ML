"""Required tracked-artifact checks; never regenerate the production model."""

import importlib.metadata
import json

import pandas as pd

from scripts.smoke_test_nev_inference import smoke_test_saved_model
from scripts.smoke_test_localhost import SAMPLE_ROWS
from src.config.paths import PROJECT_ROOT
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES
from src.prediction.nev_fault import predict_nev_faults
from src.storage.model_registry import NEV_PIPELINE_PATH, load_nev_fault_pipeline


def test_saved_nev_pipeline_inference_smoke():
    assert NEV_PIPELINE_PATH.is_file(), "Required tracked pipeline is missing; pull the complete repository, do not retrain automatically"
    output = smoke_test_saved_model()
    assert len(output) == 4
    assert set(output.fault_code).issubset({0, 1, 2, 3})


def test_user_supplied_normalized_examples_cover_all_four_classes():
    sensors = pd.DataFrame(SAMPLE_ROWS, columns=NEV_FEATURES)
    output = predict_nev_faults(load_nev_fault_pipeline(), sensors)
    assert output.fault_code.tolist() == [0, 1, 2, 3]
    assert output.fault_name.tolist() == list(NEV_CLASSES.values())


def test_model_loading_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert NEV_PIPELINE_PATH == PROJECT_ROOT / "models" / "experiment_2" / "nev_fault_pipeline.joblib"
    assert list(load_nev_fault_pipeline().feature_names_in_) == list(NEV_FEATURES)


def test_checkpoint_dependencies_match_recorded_model_versions():
    metadata = json.loads(NEV_PIPELINE_PATH.with_name("metadata.json").read_text(encoding="utf-8"))
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for package, key in (("scikit-learn", "scikit_learn"), ("pandas", "pandas"),
                         ("numpy", "numpy"), ("joblib", "joblib")):
        expected = metadata["versions"][key]
        assert f"{package}=={expected}" in requirements
        assert importlib.metadata.version(package) == expected
