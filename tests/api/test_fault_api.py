"""Local-MVP API tests; tiny synthetic fixtures, never retrain the NEV artifact."""

import csv
import io
import warnings

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from src.api.app import create_app
from src.api import telemetry
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, file_sha256
from src.preprocessing.cleaning import build_numeric_preprocessor
from src.storage.model_registry import load_nev_fault_pipeline


def sensor_values(voltage=0.35):
    return dict(zip(NEV_FEATURES, [voltage, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))


def csv_content(rows, headers=NEV_FEATURES):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


@pytest.fixture
def saved_fixture(tmp_path):
    # Fits only 16 synthetic unit-test rows; no production model/data changes.
    values = [0.1, 0.35, 0.65, 0.9]
    x = pd.DataFrame([sensor_values(value) for value in values for _ in range(4)])
    y = np.repeat([0, 1, 2, 3], 4)
    pipeline = Pipeline([
        ("preprocessor", build_numeric_preprocessor(list(NEV_FEATURES))),
        ("classifier", RandomForestClassifier(n_estimators=8, random_state=42, n_jobs=1)),
    ]).fit(x, y)
    path = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture
def client(saved_fixture):
    with TestClient(create_app(saved_fixture)) as active:
        yield active


def test_health_reports_model_and_disabled_modules(client):
    response = client.get("/health")
    assert response.status_code == 200
    status = response.json()
    assert status["status"] == "ok"
    assert status["fault_model_status"] == "ready"
    assert status["input_mode"] == "normalized_proof_of_concept"
    assert status["rul_status"] == "not_connected" and status["risk_status"] == "disabled"
    assert status["model_version"].startswith("nev-fault-")


@pytest.mark.parametrize("code,voltage", [(0, 0.1), (1, 0.35), (2, 0.65), (3, 0.9)])
def test_valid_fault_request_mapping_and_confidence(client, code, voltage):
    response = client.post("/predict-fault", json=sensor_values(voltage))
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] == code
    assert result["predicted_fault"] == NEV_CLASSES[code]
    assert 0 <= result["model_confidence"] <= 1
    assert result["model_confidence"] == result["class_probabilities"][NEV_CLASSES[code]]
    assert np.isclose(sum(result["class_probabilities"].values()), 1)
    assert "failure_probability" not in result and "risk_score" not in result and "estimated_rul" not in result


def test_request_key_order_is_safe(client):
    values = sensor_values()
    expected = client.post("/predict-fault", json=values).json()
    reordered = dict(reversed(list(values.items())))
    assert client.post("/predict-fault", json=reordered).json() == expected


@pytest.mark.parametrize("invalid", [-0.01, 1.01, "0.5", True, False, None])
def test_fault_request_rejects_invalid_numeric_values(client, invalid):
    values = sensor_values()
    values[NEV_FEATURES[0]] = invalid
    assert client.post("/predict-fault", json=values).status_code == 422


@pytest.mark.parametrize("change", ["missing", "target", "timestamp", "alias"])
def test_fault_request_rejects_missing_or_unknown_fields(client, change):
    values = sensor_values()
    if change == "missing":
        values.pop(NEV_FEATURES[0])
    elif change == "alias":
        values["voltage"] = values.pop(NEV_FEATURES[0])
    else:
        values["Fault Label" if change == "target" else "Timestamp"] = 0
    assert client.post("/predict-fault", json=values).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_returns_serializable_validation_error(client, token):
    import json
    body = json.dumps(sensor_values()).replace("0.35", token)
    response = client.post("/predict-fault", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


def test_csv_batch_schema_counts_order_and_ties(client):
    headers = tuple(reversed(NEV_FEATURES))
    rows = [[sensor_values(value)[feature] for feature in headers] for value in (0.1, 0.35, 0.65, 0.9)]
    response = client.post("/analyze-csv", files={"file": ("telemetry.csv", csv_content(rows, headers), "text/csv")})
    assert response.status_code == 200
    summary = response.json()
    assert summary["rows_analyzed"] == 4
    assert all(item == {"count": 1, "percentage": 25.0} for item in summary["classes"].values())
    assert [item["predicted_class"] for item in summary["predictions"]] == [0, 1, 2, 3]
    assert summary["dominant_abnormal_fault"] is None  # No arbitrary winner in a tie.
    assert summary["dominant_abnormal_faults"] == list(NEV_CLASSES.values())[1:]
    assert not summary["temporal_analysis"] and summary["row_order"] == "uploaded_file_order_not_time"
    assert 0 <= summary["mean_model_confidence"] <= 1


def test_csv_normal_only_does_not_invent_abnormal_fault(client):
    content = csv_content([list(sensor_values(0.1).values())] * 3)
    response = client.post("/analyze-csv", files={"file": ("data.csv", content)})
    summary = response.json()
    assert summary["classes"]["Normal"] == {"count": 3, "percentage": 100.0}
    assert summary["dominant_abnormal_fault"] is None and summary["dominant_abnormal_faults"] == []


@pytest.mark.parametrize("invalid_kind", ["missing_header", "duplicate_header", "extra_header", "empty", "header_only", "malformed", "short_row", "extra_value", "blank", "string", "nan", "infinity", "out_of_range", "wrong_encoding"])
def test_csv_invalid_upload_rejects_everything(client, invalid_kind):
    headers = list(NEV_FEATURES)
    row = list(sensor_values().values())
    if invalid_kind == "missing_header":
        content = csv_content([row[:-1]], headers[:-1])
    elif invalid_kind == "duplicate_header":
        headers[-1] = headers[0]
        content = csv_content([row], headers)
    elif invalid_kind == "extra_header":
        content = csv_content([row + [1]], headers + ["Fault Label"])
    elif invalid_kind == "empty":
        content = b""
    elif invalid_kind == "header_only":
        content = csv_content([])
    elif invalid_kind == "malformed":
        content = csv_content([]) + b'"unfinished quote'
    elif invalid_kind == "wrong_encoding":
        content = csv_content([row]).decode("utf-8").encode("cp1252")
    elif invalid_kind in ("short_row", "extra_value"):
        content = csv_content([row[:-1] if invalid_kind == "short_row" else row + [1]])
    else:
        row[0] = {"blank": "", "string": "invalid", "nan": "nan", "infinity": "inf", "out_of_range": 2}[invalid_kind]
        content = csv_content([list(sensor_values().values()), row])
    response = client.post("/analyze-csv", files={"file": ("data.csv", content)})
    assert response.status_code == 422
    assert "predictions" not in response.json()


def test_csv_supports_utf8_bom_and_floating_point_tolerance(client):
    row = list(sensor_values().values())
    row[-1] = 1.0000000000000002
    response = client.post("/analyze-csv", files={"file": ("data.csv", b"\xef\xbb\xbf" + csv_content([row]))})
    assert response.status_code == 200


def test_csv_upload_limits_and_output_truncation(client, monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_CSV_ROWS", 2)
    content = csv_content([list(sensor_values().values())] * 3)
    assert client.post("/analyze-csv", files={"file": ("data.csv", content)}).status_code == 413
    monkeypatch.setattr(telemetry, "MAX_CSV_ROWS", 20_000)
    monkeypatch.setattr(telemetry, "MAX_ROW_PREDICTIONS", 2)
    summary = client.post("/analyze-csv", files={"file": ("data.csv", content)}).json()
    assert summary["rows_analyzed"] == 3 and summary["predictions_returned"] == 2
    assert summary["predictions_truncated"] and summary["classes"]["Motor Fault"]["count"] == 3
    monkeypatch.setattr(telemetry, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/analyze-csv", files={"file": ("data.csv", content)}).status_code == 413


def test_missing_upload_is_validation_error(client):
    assert client.post("/analyze-csv").status_code == 422


def test_model_loading_never_changes_artifact(saved_fixture):
    before = file_sha256(saved_fixture)
    loaded = load_nev_fault_pipeline(saved_fixture)
    assert list(loaded.feature_names_in_) == list(NEV_FEATURES)
    assert list(loaded.classes_) == list(NEV_CLASSES)
    with TestClient(create_app(saved_fixture)) as client:
        assert client.post("/predict-fault", json=sensor_values()).status_code == 200
    assert file_sha256(saved_fixture) == before


@pytest.mark.parametrize("artifact_kind", ["missing", "corrupt", "wrong_object", "wrong_order", "wrong_classes", "wrong_classifier"])
def test_unavailable_model_does_not_train_or_fabricate_predictions(tmp_path, saved_fixture, artifact_kind):
    destination = tmp_path / "unavailable.joblib"
    if artifact_kind == "corrupt":
        destination.write_bytes(b"not a joblib artifact")
    elif artifact_kind == "wrong_object":
        joblib.dump({"not": "a pipeline"}, destination)
    elif artifact_kind in ("wrong_order", "wrong_classes", "wrong_classifier"):
        pipeline = joblib.load(saved_fixture)
        if artifact_kind == "wrong_order":
            pipeline.named_steps["preprocessor"].feature_names_in_ = np.array(list(reversed(NEV_FEATURES)))
        elif artifact_kind == "wrong_classes":
            pipeline.named_steps["classifier"].classes_ = np.array([0, 1])
        else:
            pipeline.steps[-1] = ("classifier", DummyClassifier())
        joblib.dump(pipeline, destination)
    with TestClient(create_app(destination)) as client:
        assert client.get("/health").json()["fault_model_status"] == "unavailable"
        assert client.post("/predict-fault", json=sensor_values()).status_code == 503
        assert client.get("/").status_code == 200
    if artifact_kind == "missing":
        assert not destination.exists()


def test_model_is_loaded_once_and_never_fitted_at_startup_or_request(saved_fixture, monkeypatch):
    from src.api import app as app_module

    calls = []
    def counted_load(path):
        calls.append(path)
        return load_nev_fault_pipeline(path)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Application must never fit a model")

    monkeypatch.setattr(app_module, "load_nev_fault_pipeline", counted_load)
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    with TestClient(create_app(saved_fixture)) as client:
        assert client.get("/health").json()["fault_model_status"] == "ready"
        for _ in range(3):
            assert client.post("/predict-fault", json=sensor_values()).status_code == 200
        content = csv_content([list(sensor_values().values())])
        assert client.post("/analyze-csv", files={"file": ("data.csv", content)}).status_code == 200
    assert calls == [saved_fixture]


def test_incompatible_sklearn_artifact_is_rejected(saved_fixture, monkeypatch):
    from src.storage import model_registry

    def incompatible_load(path):
        warnings.warn(InconsistentVersionWarning(
            estimator_name="RandomForestClassifier",
            current_sklearn_version="current",
            original_sklearn_version="incompatible",
        ))

    monkeypatch.setattr(model_registry.joblib, "load", incompatible_load)
    with pytest.raises(InconsistentVersionWarning):
        load_nev_fault_pipeline(saved_fixture)
    with TestClient(create_app(saved_fixture)) as client:
        assert client.get("/health").json()["fault_model_status"] == "unavailable"
        assert client.post("/predict-fault", json=sensor_values()).status_code == 503


def test_dashboard_and_backend_share_local_origin(client):
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "EV Predictive Maintenance Platform" in dashboard.text
    assert "RUL model not yet connected." in dashboard.text
    assert "Risk scoring is disabled." in dashboard.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.post("/predict-fault", json=sensor_values()).status_code == 200
    assert client.post("/predict-rul", json={}).status_code == 404


def test_dashboard_synthetic_csv_is_available_and_valid(client):
    sample = client.get("/static/synthetic_normalized_telemetry.csv")
    assert sample.status_code == 200
    parsed = telemetry.parse_telemetry_csv(sample.content)
    assert parsed.shape == (4, 7)
    assert list(parsed.columns) == list(NEV_FEATURES)
    assert "Fault Label" not in parsed.columns
    response = client.post("/analyze-csv", files={"file": ("synthetic_example.csv", sample.content)})
    assert response.status_code == 200
    assert response.json()["rows_analyzed"] == 4
