"""Experiment 2 regression tests with small generated data, not raw uploads."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.nev_fault import (
    NEV_CLASSES, NEV_FEATURES, NEV_FILES, NEV_TARGET, file_sha256,
    get_nev_xy, load_nev_datasets, validate_nev_frame, verify_supplied_split,
)
from src.eda.nev_fault import audit_nev_datasets, investigate_nev_signal
from src.evaluation.metrics import multiclass_classification_metrics
from src.prediction.nev_fault import predict_nev_faults
from src.training import nev_fault


@pytest.fixture
def nev_frames():
    rng = np.random.default_rng(42)
    training = pd.DataFrame(rng.uniform(0.05, 0.85, (80, 7)), columns=NEV_FEATURES)
    testing = pd.DataFrame(rng.uniform(0.86, 0.99, (20, 7)), columns=NEV_FEATURES)
    training[NEV_TARGET] = np.tile([0.0, 1.0, 2.0, 3.0], 20)
    testing[NEV_TARGET] = np.tile([0.0, 1.0, 2.0, 3.0], 5)
    return {"full": pd.concat([training, testing], ignore_index=True), "training": training, "testing": testing}


def test_nev_target_mapping_feature_order_and_preservation(nev_frames):
    data = nev_frames["training"]
    before = data.copy(deep=True)
    x, y = get_nev_xy(data[data.columns[::-1]])
    assert NEV_CLASSES == {0: "Normal", 1: "Motor Fault", 2: "Inverter Fault", 3: "Battery Fault"}
    assert tuple(x.columns) == NEV_FEATURES
    assert len(x.columns) == 7 and NEV_TARGET not in x
    assert pd.api.types.is_integer_dtype(y) and set(y) == {0, 1, 2, 3}
    pd.testing.assert_frame_equal(data, before)


def test_nev_roundoff_boundary_is_accepted_without_clipping(nev_frames):
    data = nev_frames["training"].copy()
    data.loc[0, NEV_FEATURES[0]] = 1.0000000000000002
    before = data.copy(deep=True)
    x, _ = get_nev_xy(data)
    assert x.iloc[0, 0] > 1
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("issue", ["missing_column", "extra_column", "non_numeric", "missing", "infinity", "out_of_range", "fractional_label", "unknown_label", "duplicate_column"])
def test_nev_schema_rejects_invalid_inputs(nev_frames, issue):
    data = nev_frames["training"].copy()
    if issue == "missing_column":
        data = data.drop(columns=NEV_FEATURES[0])
    elif issue == "extra_column":
        data["future_outcome"] = 1
    elif issue == "duplicate_column":
        data = pd.concat([data, data[[NEV_FEATURES[0]]]], axis=1)
    else:
        values = {"non_numeric": "not a number", "missing": np.nan, "infinity": np.inf,
                  "out_of_range": 200, "fractional_label": 1.5, "unknown_label": 4}
        column = NEV_TARGET if "label" in issue else NEV_FEATURES[0]
        if issue == "non_numeric":
            data[column] = data[column].astype(object)
        data.loc[0, column] = values[issue]
    with pytest.raises(ValueError):
        validate_nev_frame(data)


def test_nev_supplied_split_checks_multiplicity_and_overlap(nev_frames):
    checks = verify_supplied_split(**{key: nev_frames[key] for key in ("full", "training", "testing")})
    assert checks["training_plus_testing_equals_full_multiset"]
    assert checks["feature_only_overlap"] == checks["exact_row_overlap"] == 0
    with pytest.raises(ValueError, match="reconstruct"):
        verify_supplied_split(nev_frames["full"].iloc[:-1], nev_frames["training"], nev_frames["testing"])
    overlapping_test = pd.concat([nev_frames["testing"], nev_frames["training"].iloc[:1]])
    overlapping_full = pd.concat([nev_frames["training"], overlapping_test])
    with pytest.raises(ValueError, match="overlapping"):
        verify_supplied_split(overlapping_full, nev_frames["training"], overlapping_test)


def test_nev_loader_and_audit_leave_raw_files_unchanged(tmp_path, nev_frames):
    for name, filename in NEV_FILES.items():
        nev_frames[name].to_csv(tmp_path / filename, index=False)
    before = {name: file_sha256(tmp_path / filename) for name, filename in NEV_FILES.items()}
    loaded = load_nev_datasets(tmp_path)
    audit = audit_nev_datasets(loaded)
    assert audit["datasets"]["training"]["shape"] == [80, 8]
    assert all(value == 0 for value in audit["datasets"]["full"]["missing"].values())
    assert before == {name: file_sha256(tmp_path / filename) for name, filename in NEV_FILES.items()}


def test_nev_inference_serialization_and_reordered_columns(tmp_path, nev_frames):
    x, y = get_nev_xy(nev_frames["training"])
    pipeline = nev_fault.build_nev_candidates()["logistic_regression"].fit(x, y)
    sample = x.iloc[:8]
    expected = predict_nev_faults(pipeline, sample)
    pd.testing.assert_frame_equal(expected, predict_nev_faults(pipeline, sample[sample.columns[::-1]]))
    assert set(expected.fault_code).issubset(NEV_CLASSES)
    assert expected.fault_name.tolist() == [NEV_CLASSES[code] for code in expected.fault_code]
    assert np.allclose(expected.filter(like="probability_").sum(axis=1), 1)
    destination = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, destination)
    pd.testing.assert_frame_equal(expected, predict_nev_faults(joblib.load(destination), sample))
    with pytest.raises(ValueError, match="schema"):
        predict_nev_faults(pipeline, nev_frames["training"].iloc[:8])


def test_nev_cv_selection_and_final_fits_use_training_only(monkeypatch, nev_frames):
    original = nev_fault.build_nev_candidates

    def tiny_candidates():
        candidates = original()
        candidates["random_forest"].set_params(classifier__n_estimators=3, classifier__n_jobs=1)
        candidates["hist_gradient_boosting"].set_params(classifier__max_iter=2)
        return candidates

    monkeypatch.setattr(nev_fault, "build_nev_candidates", tiny_candidates)
    before = nev_frames["training"].copy(deep=True)
    selection = nev_fault.select_nev_model(nev_frames["training"])
    assert all(len(result["f1_macro"]["folds"]) == 5 for result in selection["cv"].values())
    expected_choice = max(selection["cv"], key=lambda name: selection["cv"][name]["f1_macro"]["mean"])
    assert selection["selected_model"] == expected_choice
    models, results = nev_fault.fit_and_evaluate_nev(nev_frames["training"], nev_frames["testing"], selection)
    lr = models["logistic_regression"]
    numeric = lr.named_steps["preprocessor"].named_transformers_["numeric"]
    x, _ = get_nev_xy(nev_frames["training"])
    assert np.allclose(numeric.named_steps["imputer"].statistics_, x.median())
    assert np.allclose(numeric.named_steps["scaler"].mean_, x.mean())
    assert numeric.named_steps["scaler"].n_samples_seen_ == 80
    assert selection["selected_model"] == expected_choice  # test scores never change selection
    assert all(np.sum(result["confusion_matrix"]) == 20 for result in results.values())
    pd.testing.assert_frame_equal(before, nev_frames["training"])


def test_nev_signal_is_training_only_and_noncausal(nev_frames):
    signal = investigate_nev_signal(nev_frames["training"])
    assert signal["scope"] == "supplied training dataset only"
    assert set(signal["associations"]) == set(NEV_FEATURES)
    assert len(signal["shallow_rule_cv_macro_f1"]["folds"]) == 5
    assert NEV_TARGET not in signal["sensor_correlation"]


def test_multiclass_metrics_probability_order_and_missing_class():
    actual = [0, 1, 2, 3]
    scores = np.eye(4)
    metrics = multiclass_classification_metrics(actual, actual, scores, [0, 1, 2, 3])
    assert metrics["accuracy"] == metrics["macro_f1"] == metrics["roc_auc_macro_ovr"] == 1
    assert metrics["confusion_matrix"] == np.eye(4, dtype=int).tolist()
    with pytest.raises(ValueError, match="order"):
        multiclass_classification_metrics(actual, actual, scores, [3, 2, 1, 0])
    assert multiclass_classification_metrics(actual[:3], actual[:3], scores[:3], actual)["roc_auc_macro_ovr"] is None
