"""Check the running MVP using user-supplied normalized test examples; no fit.

Run in a second terminal after starting uvicorn. These are application smoke
checks, not a new evaluation dataset or evidence of real-world performance.
"""

import argparse
import csv
import hashlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES

SAMPLE_ROWS = (
    (0.939, 0.200, 0.544, 0.240, 0.301, 0.582, 0.397),
    (0.805, 0.597, 0.073, 0.180, 0.921, 0.128, 0.125),
    (0.586, 0.275, 0.355, 0.398, 0.452, 0.728, 0.326),
    (0.180, 0.453, 0.393, 0.782, 0.227, 0.287, 0.515),
)


def csv_bytes(rows=SAMPLE_ROWS, headers=NEV_FEATURES):
    """Small demo fixtures only; no training datasets are uploaded."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def check_csv_uploads(client):
    """Upload actual temporary files as multipart requests; verify no mutation."""
    reordered = tuple(reversed(NEV_FEATURES))
    reordered_rows = [list(reversed(row)) for row in SAMPLE_ROWS]
    nonnumeric = [list(SAMPLE_ROWS[0])]
    nonnumeric[0][3] = "invalid"
    cases = [
        ("valid", csv_bytes(), 200),
        ("missing_column", csv_bytes([row[1:] for row in SAMPLE_ROWS], NEV_FEATURES[1:]), 422),
        ("non_numeric", csv_bytes(nonnumeric), 422),
        ("empty", b"", 422),
        ("extra_harmless", csv_bytes([list(row) + ["demo"] for row in SAMPLE_ROWS],
                                     list(NEV_FEATURES) + ["Note"]), 200),
        ("reordered", csv_bytes(reordered_rows, reordered), 200),
        ("without_target", csv_bytes(), 200),
        ("bom_whitespace", b"\xef\xbb\xbf" + csv_bytes(headers=[f"  {name}  " for name in NEV_FEATURES]), 200),
        ("ignored_wrong_target", csv_bytes([list(row) + [3 - index] for index, row in enumerate(SAMPLE_ROWS)],
                                          list(NEV_FEATURES) + ["Fault Label"]), 200),
    ]
    with TemporaryDirectory(prefix="ev-telemetry-smoke-") as directory:
        for name, content, expected_status in cases:
            path = Path(directory) / f"{name}.csv"
            path.write_bytes(content)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open("rb") as upload:
                response = client.post("/analyze-csv", files={"file": (path.name, upload, "text/csv")})
            assert response.status_code == expected_status, (name, response.text)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            result = response.json()
            if expected_status == 200:
                assert result["rows_analyzed"] == 4
                assert [row["predicted_class"] for row in result["predictions"]] == [0, 1, 2, 3]
                assert all(item == {"count": 1, "percentage": 25.0} for item in result["classes"].values())
                assert abs(result["mean_model_confidence"] - 0.9975) < 1e-12
                assert result["temporal_analysis"] is False
                if name == "extra_harmless":
                    assert result["ignored_columns"] == ["Note"]
                if name == "ignored_wrong_target":
                    assert result["ignored_columns"] == ["Fault Label"]
                if name == "bom_whitespace":
                    assert len(result["normalized_headers"]) == 7
            else:
                assert "predictions" not in result
                if name == "missing_column":
                    assert "Voltage (V)" in result["detail"]
                if name == "non_numeric":
                    assert "line 2" in result["detail"] and "Temperature (°C)" in result["detail"]
            print(f"CSV {name}: HTTP {expected_status}; upload unchanged")


def smoke_test_localhost(base_url="http://127.0.0.1:8012"):
    """Verify health, all four classes, validation, assets and CSV over HTTP."""
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["fault_model_status"] == "ready", health.json()
        assert health.json()["rul_status"] == "not_connected"
        assert health.json()["risk_status"] == "disabled"
        print("Health: model ready; RUL/risk disabled")
        for code, values in enumerate(SAMPLE_ROWS):
            payload = dict(zip(NEV_FEATURES, values))
            response = client.post("/predict-fault", json=payload)
            response.raise_for_status()
            result = response.json()
            assert result["predicted_class"] == code, result
            assert result["predicted_fault"] == NEV_CLASSES[code]
            scores = result["class_probabilities"]
            assert set(scores) == set(NEV_CLASSES.values())
            assert all(0 <= probability <= 1 for probability in scores.values())
            assert abs(sum(scores.values()) - 1) < 1e-12
            assert 0 <= result["model_confidence"] <= 1
            assert result["model_confidence"] == scores[NEV_CLASSES[code]]
            assert result["input_mode"] == "normalized_proof_of_concept"
            assert not {"estimated_rul", "risk_score", "failure_probability"} & result.keys()
            reordered = client.post("/predict-fault", json=dict(reversed(list(payload.items()))))
            reordered.raise_for_status()
            assert reordered.json() == result
            print(f"{code} = {result['predicted_fault']}: class confidence {result['model_confidence']:.3f}")

        invalid = dict(zip(NEV_FEATURES, SAMPLE_ROWS[0]))
        invalid[NEV_FEATURES[0]] = 2
        assert client.post("/predict-fault", json=invalid).status_code == 422
        invalid.pop(NEV_FEATURES[0])
        assert client.post("/predict-fault", json=invalid).status_code == 422
        print("Invalid and missing fields: HTTP 422")

        dashboard = client.get("/")
        dashboard.raise_for_status()
        for text in ("EVision | AI-Powered EV Predictive Maintenance", "Prototype input mode",
                     "Awaiting vehicle analysis", "Download Sample CSV",
                     "RUL model not yet connected.", "Risk scoring is disabled."):
            assert text in dashboard.text
        for asset, expected_mime in (("app.js?v=evision-1", "text/javascript"),
                                     ("styles.css?v=evision-1", "text/css"),
                                     ("evision-car.webp", "image/webp"),
                                     ("synthetic_normalized_telemetry.csv", "text/csv")):
            response = client.get(f"/static/{asset}")
            response.raise_for_status()
            actual_mime = response.headers["content-type"].split(";")[0]
            assert actual_mime == expected_mime or (expected_mime == "text/javascript" and actual_mime == "application/javascript")
            print(f"Asset {asset}: HTTP {response.status_code}; {actual_mime}")
        print("Dashboard and required static assets: HTTP 200")

        # Download the UI sample too, then exercise all required upload cases.
        sample = client.get("/static/synthetic_normalized_telemetry.csv")
        response = client.post("/analyze-csv", files={"file": ("downloaded_sample.csv", sample.content, "text/csv")})
        response.raise_for_status()
        summary = response.json()
        assert summary["rows_analyzed"] == 4
        assert [row["predicted_class"] for row in summary["predictions"]] == [0, 1, 2, 3]
        assert all(item == {"count": 1, "percentage": 25.0} for item in summary["classes"].values())
        assert 0 <= summary["mean_model_confidence"] <= 1
        assert summary["temporal_analysis"] is False
        print("CSV: 4 rows; one prediction per class, 25% each; no temporal claim")
        check_csv_uploads(client)
        print("All application smoke checks passed; no model training performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    smoke_test_localhost(parser.parse_args().base_url)
