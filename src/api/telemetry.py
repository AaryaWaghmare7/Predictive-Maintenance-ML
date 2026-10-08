"""Bounded, in-memory CSV validation and shared prediction serialization."""

from __future__ import annotations

import csv
import io
import math

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.api.schemas import ClassSummary, FaultPrediction, TelemetrySummary
from src.data.nev_fault import NEV_CLASSES, NEV_FEATURES, NORMALIZED_RANGE_TOLERANCE, validate_nev_frame
from src.prediction.nev_fault import predict_nev_faults

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 20_000
MAX_ROW_PREDICTIONS = 1_000


class TelemetryValidationError(ValueError):
    """Bad user CSV; the entire upload is rejected without partial prediction."""


class TelemetryLimitError(TelemetryValidationError):
    """The upload exceeds a documented local-MVP resource limit."""


def parse_telemetry_csv(content: bytes) -> pd.DataFrame:
    """Require UTF-8/BOM, seven unique headers and finite normalized values.

    No column renaming, row dropping, physical-unit scaling or persistence.
    Extra target/timestamp/ID columns are rejected, not used as features.
    """
    if len(content) > MAX_UPLOAD_BYTES:
        raise TelemetryLimitError("CSV exceeds the 5 MiB upload limit")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise TelemetryValidationError("CSV must be encoded as UTF-8 (UTF-8 BOM is supported)") from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = next(reader, None)
        if headers is None or len(headers) != len(NEV_FEATURES) or set(headers) != set(NEV_FEATURES):
            raise TelemetryValidationError("CSV must have exactly the seven unique Experiment 2 feature headers")
        rows = []
        for record in reader:
            if len(rows) >= MAX_CSV_ROWS:
                raise TelemetryLimitError("CSV exceeds the 20,000-row limit")
            if len(record) != len(headers):
                raise TelemetryValidationError(f"CSV line {reader.line_num}: expected seven values")
            try:
                row = [float(cell) for cell in record]
                # Cheap per-row check, followed by the shared frame validator.
                if not all(math.isfinite(value) and -NORMALIZED_RANGE_TOLERANCE <= value
                           <= 1 + NORMALIZED_RANGE_TOLERANCE for value in row):
                    raise ValueError("Nonfinite or out-of-range sensor value")
            except (ValueError, TypeError) as error:
                raise TelemetryValidationError(
                    f"CSV line {reader.line_num}: values must be finite numeric inputs normalized to [0,1]"
                ) from error
            rows.append(row)
    except csv.Error as error:
        raise TelemetryValidationError("CSV quoting or structure is malformed") from error
    if not rows:
        raise TelemetryValidationError("CSV contains no sensor observations")
    frame = pd.DataFrame(rows, columns=headers)
    validate_nev_frame(frame, require_target=False)
    return frame.loc[:, list(NEV_FEATURES)]


def make_prediction(row: pd.Series, model_version: str) -> FaultPrediction:
    """Expose predicted-class probability, not a future failure/risk score."""
    code = int(row["fault_code"])
    scores = {name: float(row[f"probability_{label}"]) for label, name in NEV_CLASSES.items()}
    if not np.isfinite(list(scores.values())).all() or not np.isclose(sum(scores.values()), 1):
        raise ValueError("Model returned invalid class probabilities")
    return FaultPrediction(
        predicted_class=code, predicted_fault=NEV_CLASSES[code],
        model_confidence=scores[NEV_CLASSES[code]], class_probabilities=scores,
        status="normal" if code == 0 else "fault_detected", model_version=model_version,
    )


def analyze_telemetry(content: bytes, pipeline: Pipeline, model_version: str) -> TelemetrySummary:
    """Validate all rows, predict once, and summarize independent observations."""
    sensors = parse_telemetry_csv(content)
    output = predict_nev_faults(pipeline, sensors)
    counts = output["fault_code"].value_counts()
    classes = {name: ClassSummary(count=int(counts.get(code, 0)),
                                percentage=float(counts.get(code, 0)) / len(output) * 100)
               for code, name in NEV_CLASSES.items()}
    abnormal_max = max(classes[NEV_CLASSES[code]].count for code in (1, 2, 3))
    dominant = [NEV_CLASSES[code] for code in (1, 2, 3)
                if abnormal_max > 0 and classes[NEV_CLASSES[code]].count == abnormal_max]
    # Validate every model output, even when the returned row list is capped.
    predictions = [make_prediction(row, model_version) for _, row in output.iterrows()]
    returned = predictions[:MAX_ROW_PREDICTIONS]
    return TelemetrySummary(
        rows_analyzed=len(output), classes=classes,
        dominant_abnormal_fault=dominant[0] if len(dominant) == 1 else None,
        dominant_abnormal_faults=dominant,
        mean_model_confidence=float(np.mean([row.model_confidence for row in predictions])),
        model_version=model_version, predictions_returned=len(returned),
        predictions_truncated=len(output) > len(returned), predictions=returned,
    )
