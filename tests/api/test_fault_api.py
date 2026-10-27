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
    assert client.get("/ready").status_code == 200


def test_readiness_fails_when_artifact_is_unavailable(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as active:
        assert active.get("/health").status_code == 200
        assert active.get("/ready").status_code == 503


@pytest.mark.parametrize("chunked", [False, True])
def test_request_limit_precedes_multipart_inference(client, monkeypatch, chunked):
    from src.api import app as app_module
    monkeypatch.setattr(app_module, "MAX_REQUEST_BYTES", 200)
    body = b'--demo\r\nContent-Disposition: form-data; name="file"; filename="test.csv"\r\n\r\n' + b"x" * 300 + b"\r\n--demo--\r\n"
    response = client.post("/analyze-csv", content=iter([body[:100], body[100:]]) if chunked else body,
                           headers={"Content-Type": "multipart/form-data; boundary=demo"})
    assert response.status_code == 413
    assert "predictions" not in response.json()


def test_basic_security_headers_and_manual_body_limit(client):
    response = client.get("/")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert client.post("/predict-fault", content=b" " * 17000,
                       headers={"Content-Type": "application/json"}).status_code == 413


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


@pytest.mark.parametrize("invalid_kind", ["missing_header", "duplicate_header", "trimmed_duplicate", "blank_header", "duplicate_extra", "empty", "header_only", "malformed", "short_row", "extra_value", "blank", "blank_line", "string", "nan", "infinity", "out_of_range", "wrong_encoding"])
def test_csv_invalid_upload_rejects_everything(client, invalid_kind):
    headers = list(NEV_FEATURES)
    row = list(sensor_values().values())
    if invalid_kind == "missing_header":
        content = csv_content([row[:-1]], headers[:-1])
    elif invalid_kind == "duplicate_header":
        headers[-1] = headers[0]
        content = csv_content([row], headers)
    elif invalid_kind == "trimmed_duplicate":
        headers[-1] = "  " + headers[0] + "  "
        content = csv_content([row], headers)
    elif invalid_kind == "blank_header":
        content = csv_content([row + ["ignored"]], headers + ["  "])
    elif invalid_kind == "duplicate_extra":
        content = csv_content([row + [1, 2]], headers + ["Note", " Note "])
    elif invalid_kind == "empty":
        content = b""
    elif invalid_kind == "header_only":
        content = csv_content([])
    elif invalid_kind == "blank_line":
        content = csv_content([row]) + b"\n"
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


def test_csv_missing_column_error_names_every_missing_sensor(client):
    headers = NEV_FEATURES[1:-1]
    content = csv_content([[0.5] * len(headers)], headers)
    response = client.post("/analyze-csv", files={"file": ("missing.csv", content)})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "missing required sensor columns" in detail
    assert "Voltage (V)" in detail and "Humidity (%)" in detail


def test_csv_sensor_error_identifies_line_and_feature(client):
    row = list(sensor_values().values())
    row[3] = "not a number"
    response = client.post("/analyze-csv", files={"file": ("invalid.csv", csv_content([row]))})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "line 2" in detail and "Temperature (°C)" in detail and "[0,1]" in detail


def test_csv_bom_whitespace_and_reordering_do_not_modify_upload(client, tmp_path):
    headers = tuple(reversed(NEV_FEATURES))
    rows = [[sensor_values(value)[feature] for feature in headers] for value in (0.1, 0.35, 0.65, 0.9)]
    original_headers = [f"  {header} \t" for header in headers]
    path = tmp_path / "bom_and_spaces.csv"
    path.write_bytes(b"\xef\xbb\xbf" + csv_content(rows, original_headers))
    before = file_sha256(path)
    with path.open("rb") as upload:
        response = client.post("/analyze-csv", files={"file": (path.name, upload, "text/csv")})
    assert response.status_code == 200
    summary = response.json()
    assert [item["predicted_class"] for item in summary["predictions"]] == [0, 1, 2, 3]
    assert summary["normalized_headers"] == dict(zip(original_headers, headers))
    assert summary["ignored_columns"] == []
    assert file_sha256(path) == before
    parsed = telemetry.parse_telemetry_csv(path.read_bytes())
    assert list(parsed.columns) == list(NEV_FEATURES)
    assert parsed.iloc[0].to_dict() == sensor_values(0.1)


def test_csv_extra_columns_are_reported_but_never_model_inputs(client, monkeypatch):
    # Replaces the old "extra_header => reject" policy with the requested
    # allow-and-ignore policy. Deliberately wrong labels must not affect output.
    rows = [list(sensor_values(value).values()) for value in (0.1, 0.35, 0.65, 0.9)]
    baseline = client.post("/analyze-csv", files={"file": ("sensors.csv", csv_content(rows))}).json()
    observed_columns = []
    original_predict = telemetry.predict_nev_faults

    def observed_predict(pipeline, sensors):
        observed_columns.append(list(sensors.columns))
        return original_predict(pipeline, sensors)

    monkeypatch.setattr(telemetry, "predict_nev_faults", observed_predict)
    extras = ["Fault Label", "Note", "Timestamp", "Vehicle ID"]
    content = csv_content([row + [3 - index, "not numeric", "unknown", "EV-1"]
                           for index, row in enumerate(rows)], list(NEV_FEATURES) + extras)
    response = client.post("/analyze-csv", files={"file": ("metadata.csv", content)})
    assert response.status_code == 200
    summary = response.json()
    assert observed_columns == [list(NEV_FEATURES)]
    assert summary["ignored_columns"] == extras
    assert summary["predictions"] == baseline["predictions"]
    assert summary["classes"] == baseline["classes"]
    assert summary["mean_model_confidence"] == baseline["mean_model_confidence"]
    assert summary["temporal_analysis"] is False


def test_csv_invalid_later_row_never_runs_partial_inference(client, monkeypatch):
    def forbidden_predict(*args):
        raise AssertionError("The entire CSV must be valid before any inference")

    monkeypatch.setattr(telemetry, "predict_nev_faults", forbidden_predict)
    row = list(sensor_values().values())
    content = csv_content([row, row[:-1] + ["bad"]])
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
    assert "EVision | AI-Powered EV Predictive Maintenance" in dashboard.text
    assert "RUL model not yet connected." in dashboard.text
    assert "Risk scoring is disabled." in dashboard.text
    assert "Awaiting vehicle analysis" in dashboard.text
    for name in ("Normal", "Motor Fault", "Inverter Fault", "Battery Fault"):
        assert f"Load {name} Sample" in dashboard.text
    assert "Experiment 2 test-set performance" in dashboard.text
    assert "Fault Label is not required." in dashboard.text
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


def test_charcoal_dashboard_keeps_unverified_modules_pending(client):
    page = client.get("/").text
    styles = client.get("/static/styles.css").text
    assert "Pending verified dataset" in page
    assert "Pending validated evidence" in page
    assert "No RUL estimate or degradation curve is available" in page
    assert "No risk level, probability or automatic maintenance recommendation" in page
    assert 'type="button" disabled>Estimate RUL' in page
    assert 'type="button" disabled>Calculate risk' in page
    assert "color-scheme: dark" in styles
    assert "background: #1D1D1D" in styles
    assert "prefers-reduced-motion: reduce" in styles
    assert "[aria-busy=true] .primary-button::before" in styles
    assert "https://" not in styles  # Fonts/styles work offline.
    health = client.get("/health").json()
    assert health["rul_status"] == "not_connected" and health["risk_status"] == "disabled"


def test_automotive_dashboard_navigation_and_genuine_data_contract(client):
    from html.parser import HTMLParser

    class PageLinks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
            self.anchors = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            if tag == "a" and attrs.get("href", "").startswith("#"):
                self.anchors.append(attrs["href"][1:])

    page = client.get("/").text
    parsed = PageLinks()
    parsed.feed(page)
    assert len(parsed.ids) == len(set(parsed.ids))
    assert set(parsed.anchors).issubset(parsed.ids)
    for name in ("Home", "Diagnostics", "Analytics", "Models", "Reports", "Documentation"):
        assert f">{name}</a>" in page
    assert "Predict. <span>Prevent.</span>" in page
    assert "not live fleet statistics" in page
    assert 'id="donut-total">—</strong>' in page
    for metric in ("99.42", "99.21", "99.23"):
        assert metric in page
    assert client.get("/static/evision-car.webp").headers["content-type"] == "image/webp"
    assert 'aria-label="EVision home"' in page
    assert 'id="motion-toggle"' in page
    assert "EV Predictive Maintenance Platform" not in page
    assert client.get("/openapi.json").json()["info"]["title"] == "EVision"


def test_frontend_assets_are_served_with_correct_mime_types(client):
    import re

    page = client.get("/").text
    assets = re.findall(r'(?:href|src)="(/static/[^\"]+)"', page)
    assert any("styles.css?" in url for url in assets)
    assert any("app.js?" in url for url in assets)
    for url in assets:
        response = client.get(url)
        assert response.status_code == 200, url
        content_type = response.headers["content-type"].split(";")[0]
        if ".css" in url:
            assert content_type == "text/css"
        elif ".js" in url:
            assert content_type in {"text/javascript", "application/javascript"}
        elif ".png" in url:
            assert content_type == "image/png"
        elif ".webp" in url:
            assert content_type == "image/webp"
    assert 'id="local-preview-guard"' in page
    # SVG dimensions stay bounded even if a stylesheet cannot be loaded.
    assert 'class="icon-library"' in page and 'width="0" height="0"' in page
    for icon in re.findall(r'<svg class="icon"[^>]*>', page):
        assert 'width="24" height="24"' in icon
