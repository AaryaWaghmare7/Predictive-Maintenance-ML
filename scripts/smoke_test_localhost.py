"""Check the running MVP using user-supplied normalized test examples; no fit.

Run in a second terminal after starting uvicorn. These are application smoke
checks, not a new evaluation dataset or evidence of real-world performance.
"""

import argparse
import csv
import io
from pathlib import Path
import sys

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


def smoke_test_localhost(base_url="http://127.0.0.1:8010"):
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
        for text in ("EV Predictive Maintenance Platform", "Prototype mode:",
                     "RUL model not yet connected.", "Risk scoring is disabled."):
            assert text in dashboard.text
        for asset in ("app.js", "styles.css", "synthetic_normalized_telemetry.csv"):
            client.get(f"/static/{asset}").raise_for_status()
        print("Dashboard and required static assets: HTTP 200")

        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(NEV_FEATURES)
        writer.writerows(SAMPLE_ROWS)
        response = client.post("/analyze-csv", files={
            "file": ("normalized_smoke_examples.csv", stream.getvalue().encode("utf-8"), "text/csv"),
        })
        response.raise_for_status()
        summary = response.json()
        assert summary["rows_analyzed"] == 4
        assert [row["predicted_class"] for row in summary["predictions"]] == [0, 1, 2, 3]
        assert all(item == {"count": 1, "percentage": 25.0} for item in summary["classes"].values())
        assert 0 <= summary["mean_model_confidence"] <= 1
        assert summary["temporal_analysis"] is False
        print("CSV: 4 rows; one prediction per class, 25% each; no temporal claim")
        print("All application smoke checks passed; no model training performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    smoke_test_localhost(parser.parse_args().base_url)
