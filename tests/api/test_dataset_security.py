"""Upload, access and guidance regressions using the unchanged saved artifact."""
import csv
import io
import sqlite3

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.datasets import read_dataset
from src.api.telemetry import TelemetryValidationError
from src.data.nev_fault import NEV_FEATURES
from scripts.smoke_test_localhost import SAMPLE_ROWS


def csv_bytes(rows=SAMPLE_ROWS, labels=None, delimiter=","):
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter=delimiter)
    writer.writerow([*NEV_FEATURES, *(["Fault Label"] if labels is not None else [])])
    writer.writerows([*row, *([labels[i]] if labels is not None else [])] for i, row in enumerate(rows))
    return stream.getvalue().encode()


@pytest.fixture
def client():
    with TestClient(create_app()) as current:
        yield current


def upload(client, content=None, **data):
    return client.post("/analyze-dataset", files={"file": ("ignored_filename.csv", csv_bytes() if content is None else content)}, data=data)


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_inspection_does_not_predict_and_detects_delimiters(client, delimiter):
    result = upload(client, csv_bytes(delimiter=delimiter)).json()
    assert result["inspection"]["compatible"]
    assert result["inspection"]["rows"] == 4
    assert result["inspection"]["delimiter"] == delimiter
    assert "prediction" not in result


def test_explicit_confirmation_required_and_prediction_not_evaluation(client):
    assert upload(client, mode="predict").status_code == 422
    result = upload(client, mode="predict", normalized="true").json()
    assert result["evaluation"] is None
    assert [p["predicted_class"] for p in result["prediction"]["predictions"]] == [0, 1, 2, 3]


def test_labeled_evaluation_and_label_is_not_feature(client):
    assert upload(client, mode="evaluate", normalized="true").status_code == 422
    result = upload(client, csv_bytes(labels=[0, 1, 2, 3]), mode="evaluate", normalized="true", labels_confirmed="true").json()
    metrics = result["evaluation"]
    assert metrics["accuracy"] == 1 and metrics["macro_f1"] == 1
    assert metrics["rows_evaluated"] == 4 and metrics["rows_rejected"] == 0
    assert metrics["confusion_matrix"] == [[int(i == j) for j in range(4)] for i in range(4)]
    assert metrics["independent_test_performance"] is False
    assert metrics["macro_average_precision"] is not None
    assert metrics["multiclass_brier_sum"] >= 0
    wrong = upload(client, csv_bytes(labels=[3, 2, 1, 0]), mode="evaluate", normalized="true", labels_confirmed="true").json()
    assert wrong["evaluation"]["accuracy"] == 0
    assert wrong["prediction"] == result["prediction"]


def test_missing_evaluation_classes_give_undefined_auc_not_fake_score(client):
    result = upload(client, csv_bytes([SAMPLE_ROWS[0]], labels=[0]), mode="evaluate", normalized="true", labels_confirmed="true").json()
    assert result["evaluation"]["roc_auc_macro_ovr"] is None
    assert result["evaluation"]["macro_average_precision"] is None


@pytest.mark.parametrize("bad", ["", "NaN", "inf", "word", "1.5"])
def test_quality_report_distinguishes_bad_cells(client, bad):
    rows = [list(SAMPLE_ROWS[0])]; rows[0][0] = bad
    result = upload(client, csv_bytes(rows)).json()["inspection"]
    assert not result["compatible"] and result["issues"][0]["observation"] == 1
    assert result["issues"][0]["column"] == NEV_FEATURES[0]
    assert upload(client, csv_bytes(rows), mode="predict", normalized="true").status_code == 422


def test_mapping_is_explicit_and_never_converts_units(client):
    import json
    data = csv_bytes().decode().replace("Voltage (V)", "confirmed_voltage").encode()
    assert upload(client, data).json()["inspection"]["missing_features"] == [NEV_FEATURES[0]]
    result = upload(client, data, mapping=json.dumps({NEV_FEATURES[0]: "confirmed_voltage"})).json()
    assert result["inspection"]["compatible"]
    assert upload(client, mapping=json.dumps({NEV_FEATURES[0]: "Fault Label"})).status_code == 422


@pytest.mark.parametrize("labels", [[0, 1, 2, 4], [0, 1, 2, "wrong"], [0, 1, 2, 1.5]])
def test_bad_labels_never_get_evaluation_metrics(client, labels):
    assert upload(client, csv_bytes(labels=labels), mode="evaluate", normalized="true", labels_confirmed="true").status_code == 422


def test_files_are_independent_and_duplicates_reported(client):
    first = upload(client, mode="predict", normalized="true").json()
    second = upload(client, csv_bytes([SAMPLE_ROWS[1]]*2), mode="predict", normalized="true").json()
    assert first["prediction"]["rows_analyzed"] == 4
    assert second["prediction"]["rows_analyzed"] == 2
    assert second["inspection"]["duplicate_rows"] == 1
    assert second["prediction"]["classes"]["Motor Fault"]["percentage"] == 100


def sqlite_bytes(tmp_path):
    path = tmp_path / "sensors.sqlite"
    with sqlite3.connect(path) as db:
        names = ",".join('"'+name+'" REAL' for name in NEV_FEATURES)
        db.execute(f'CREATE TABLE observations ({names})')
        db.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?)", SAMPLE_ROWS)
        db.execute("CREATE VIEW dangerous AS SELECT load_extension('/not/allowed')")
        db.execute("CREATE TRIGGER no_writes AFTER INSERT ON observations BEGIN DELETE FROM observations; END")
    return path.read_bytes()


def test_sqlite_explicit_table_readonly_and_injection_rejected(client, tmp_path):
    content = sqlite_bytes(tmp_path)
    inspection = upload(client, content).json()["inspection"]
    assert inspection["tables"] == ["observations"] and not inspection["compatible"]
    result = upload(client, content, table="observations", mode="predict", normalized="true").json()
    assert result["prediction"]["rows_analyzed"] == 4
    for table in ("dangerous", "observations; DROP TABLE observations", "sqlite_schema", "../../secrets"):
        assert upload(client, content, table=table).status_code == 422
    assert upload(client, content, table="observations").json()["inspection"]["rows"] == 4
    assert (tmp_path / "sensors.sqlite").read_bytes() == content


def test_binary_sql_and_column_limits_reject_safely(client):
    for content in (b"SQLite format 3\x00bad", b"SELECT * FROM secret;", b"\x00binary"):
        assert upload(client, content).status_code in (200, 422)
        result = upload(client, content).json()
        assert "prediction" not in result
    assert upload(client, (",".join(f"c{i}" for i in range(101))+"\n").encode()).status_code == 413


def test_sqlite_generated_and_virtual_tables_are_not_executed(tmp_path):
    path = tmp_path / "untrusted.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE generated (x REAL, y REAL GENERATED ALWAYS AS (x+1))")
        db.execute("CREATE VIRTUAL TABLE virtual USING fts5(text)")
    content = path.read_bytes()
    with pytest.raises(TelemetryValidationError):
        read_dataset(content, "virtual")
    with pytest.raises(TelemetryValidationError, match="Generated/hidden"):
        read_dataset(content, "generated")


def test_known_training_file_cannot_be_presented_as_evaluation(client):
    from src.data.nev_fault import NEV_DATA_DIR
    path = NEV_DATA_DIR / "NEV_fault_training_dataset.csv"
    if not path.is_file():
        pytest.skip("Optional original training CSV unavailable")
    response = upload(client, path.read_bytes(), mode="evaluate", normalized="true", labels_confirmed="true")
    assert response.status_code == 422 and "training-containing" in response.json()["detail"]


@pytest.mark.parametrize("code", range(4))
def test_guidance_has_observations_safety_and_no_fake_rul_or_risk(client, code):
    response = client.post("/predict-fault", json=dict(zip(NEV_FEATURES, SAMPLE_ROWS[code]))).json()
    guidance = response["guidance"]
    assert response["predicted_class"] == code
    assert guidance["version"] and guidance["inspection_steps"] and guidance["safety"]
    assert guidance["rul"]["status"] == guidance["failure_risk"]["status"] == "unavailable"
    assert guidance["maintenance_priority"]["category"] == ("routine_review" if code == 0 else "technician_review_requested")


def test_access_headers_origin_content_type_traversal_and_cors(client):
    page = client.get("/")
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "script-src 'self' 'sha256-" in page.headers["content-security-policy"]
    assert client.post("/predict-fault", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/predict-fault", content="{}", headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 400
    for path in ("/static/../app.py", "/static/%2e%2e/app.py", "/static/.env", "/static/nev_fault_pipeline.joblib", "/admin"):
        assert client.get(path).status_code == 404
    assert "access-control-allow-origin" not in client.options("/predict-fault", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}).headers


def test_exact_generated_swagger_initializer_hash_is_allowed(client):
    import re
    import base64
    import hashlib
    response = client.get("/docs")
    assert response.status_code == 200
    inline = re.findall(r"<script>([\s\S]*?)</script>", response.text)[0]
    digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
    assert "'sha256-" + digest + "'" in response.headers["content-security-policy"]


def test_rate_and_concurrency_limits(client):
    policy = client.app.state.access_policy
    policy.manual_limit = 1
    assert client.post("/predict-fault", json=dict(zip(NEV_FEATURES, SAMPLE_ROWS[0]))).status_code == 200
    response = client.post("/predict-fault", json={})
    assert response.status_code == 429 and response.headers["Retry-After"] == "60"
    policy.active_uploads = 2
    assert upload(client).status_code == 429
    policy.active_uploads = 0
    assert upload(client).status_code == 200


def test_production_requires_secret_auth_and_hides_docs(monkeypatch):
    monkeypatch.setenv("EVISION_MODE", "production")
    monkeypatch.delenv("EVISION_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Production requires"):
        create_app()
    token = "unit-test-only-not-a-real-secret-0123456789"
    monkeypatch.setenv("EVISION_API_TOKEN", token)
    monkeypatch.setenv("EVISION_ALLOWED_HOSTS", "testserver")
    with TestClient(create_app()) as client:
        assert client.app.debug is False
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.post("/predict-fault", json={}).status_code == 401
        assert upload(client).status_code == 401
        assert client.get("/health").json()["authentication_required"] is True
        payload = dict(zip(NEV_FEATURES, SAMPLE_ROWS[0]))
        assert client.post("/predict-fault", json=payload, headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.post("/predict-fault", json=payload, headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200 and token not in response.text
        assert "strict-transport-security" not in client.get("/health").headers
        assert client.get("https://testserver/health").headers["strict-transport-security"] == "max-age=31536000"


def test_checksum_is_checked_before_canonical_joblib_deserialization(tmp_path, monkeypatch):
    from src.storage import model_registry
    path = tmp_path / "untrusted.joblib"
    path.write_bytes(b"not the approved artifact")
    monkeypatch.setattr(model_registry, "NEV_PIPELINE_PATH", path)
    def forbidden_load(*args):
        raise AssertionError("Must reject bad bytes before deserializing")
    monkeypatch.setattr(model_registry.joblib, "load", forbidden_load)
    with pytest.raises(ValueError, match="checksum"):
        model_registry.load_nev_fault_pipeline(path)


def test_labels_validate_before_any_inference(client, monkeypatch):
    from src.api import datasets
    def forbidden_predict(*args):
        raise AssertionError("Invalid evaluation labels must not run inference")
    monkeypatch.setattr(datasets, "predict_nev_faults", forbidden_predict)
    assert upload(client, csv_bytes(labels=[0,1,2,99]), mode="evaluate", normalized="true", labels_confirmed="true").status_code == 422
