"""Bounded upload inspection; explicit schema/scale confirmation, never training."""

import csv
import io
import json
import hashlib
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from src.api.telemetry import (MAX_UPLOAD_BYTES, MAX_CSV_ROWS, MAX_COLUMNS,
    TelemetryLimitError, TelemetryValidationError, parse_telemetry_csv, summarize_predictions)
from src.data.nev_fault import NEV_FEATURES, NEV_TARGET, NEV_CLASSES
from src.evaluation.metrics import multiclass_classification_metrics, multiclass_probability_diagnostics
from src.prediction.nev_fault import predict_nev_faults
from src.eda.dashboard import build_dashboard_data, load_dashboard_reference


def read_dataset(content: bytes, table: str = "") -> tuple[pd.DataFrame, dict]:
    """Read CSV or an in-memory, query-only SQLite ordinary table. No SQL uploads."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise TelemetryLimitError("Dataset exceeds the 5 MiB upload limit")
    if content.startswith(b"SQLite format 3\x00"):
        if not hasattr(sqlite3.Connection, "deserialize"):
            raise TelemetryValidationError("This Python lacks safe in-memory SQLite import; export the table as CSV")
        connection = sqlite3.connect(":memory:")
        try:
            connection.deserialize(content)
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA cell_size_check=ON")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_UPLOAD_BYTES)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, MAX_COLUMNS)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 20)
            connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            deadline = time.monotonic() + 2
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            if hasattr(connection, "setconfig"):
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_VIEW, False)
            objects = connection.execute("SELECT name,type,sql FROM sqlite_schema LIMIT 101").fetchall()
            if len(objects) > 100:
                raise TelemetryLimitError("SQLite schema exceeds 100 objects")
            tables = [name for name, kind, sql in objects if kind == "table" and
                      not name.startswith("sqlite_") and sql and
                      sql.lstrip().upper().startswith("CREATE TABLE")]
            # Never select views, virtual tables, system tables or user SQL.
            if not table:
                return pd.DataFrame(), {"format": "sqlite", "tables": tables,
                                        "selection_required": True}
            if table not in tables:
                raise TelemetryValidationError("Select an ordinary table from the inspection's allowed table names")
            quoted = '"' + table.replace('"', '""') + '"'
            columns = connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
            if any(column[6] for column in columns):
                raise TelemetryValidationError("Generated/hidden SQLite columns are unsupported; export ordinary sensor columns as CSV")
            connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ) else sqlite3.SQLITE_DENY)
            cursor = connection.execute(f"SELECT * FROM {quoted} LIMIT ?", (MAX_CSV_ROWS + 1,))
            records = cursor.fetchall()
            if len(records) > MAX_CSV_ROWS:
                raise TelemetryLimitError("Dataset exceeds the 20,000-row limit")
            return pd.DataFrame(records, columns=[col[0] for col in cursor.description]), {
                "format": "sqlite", "table": table, "tables": tables,
                "row_order": "database_return_order_not_time", "selection_required": False}
        except sqlite3.Error:
            raise TelemetryValidationError("SQLite database is malformed, unsupported, or exceeded its query budget; export a plain CSV") from None
        finally:
            connection.close()
    try:
        text = content.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        headers = next(reader, [])
        if not headers:
            raise TelemetryValidationError("CSV is empty; provide headers and sensor observations")
        if len(headers) > MAX_COLUMNS or any(len(h) > 128 for h in headers):
            raise TelemetryLimitError("Use at most 100 columns and 128-character headers")
        rows, line_numbers = [], []
        for row in reader:
            if len(rows) >= MAX_CSV_ROWS:
                raise TelemetryLimitError("Dataset exceeds the 20,000-row limit")
            if len(row) != len(headers):
                raise TelemetryValidationError(f"CSV line {reader.line_num}: expected {len(headers)} cells; found {len(row)}")
            rows.append(row)
            line_numbers.append(reader.line_num)
        frame = pd.DataFrame(rows, columns=headers)
        frame.attrs["csv_lines"] = line_numbers
        return frame, {"format": "csv", "delimiter": delimiter,
            "selection_required": False, "row_order": "uploaded_file_order_not_time"}
    except (UnicodeError, csv.Error, ValueError) as error:
        if isinstance(error, TelemetryValidationError):
            raise
        raise TelemetryValidationError("Use UTF-8 CSV (comma, semicolon, tab or pipe) or a SQLite database; SQL dumps and other binary files are unsupported") from None


def inspect_dataset(content: bytes, table: str = "", mapping: str = "{}"):
    frame, source = read_dataset(content, table)
    if source.get("selection_required"):
        return {**source, "compatible": False, "reason": "Choose a table explicitly"}, None
    try:
        mapping = json.loads(mapping)
        if (not isinstance(mapping, dict) or not set(mapping).issubset(NEV_FEATURES)
                or not all(isinstance(v, str) for v in mapping.values())):
            raise ValueError
    except (ValueError, TypeError):
        raise TelemetryValidationError("Mapping must be a JSON object: canonical sensor name to uploaded column name") from None
    originals = list(frame.columns)
    trimmed = [str(c).strip() for c in originals]
    if len(trimmed) > MAX_COLUMNS or any(len(c) > 128 for c in trimmed):
        raise TelemetryLimitError("Use at most 100 columns and 128-character headers")
    if len(set(trimmed)) != len(trimmed) or any(not c for c in trimmed):
        raise TelemetryValidationError("Column names must be nonblank and unique after trimming whitespace")
    frame.columns = trimmed
    if any(v not in trimmed for v in mapping.values()):
        raise TelemetryValidationError("Every mapped source column must exist")
    if any(target in trimmed and source != target for target, source in mapping.items()):
        raise TelemetryValidationError("Mapping conflicts with an existing canonical sensor column; remove ambiguity explicitly")
    selected = [mapping.get(feature, feature) for feature in NEV_FEATURES]
    if len(set(selected)) != len(selected) or NEV_TARGET in selected:
        raise TelemetryValidationError("Each sensor needs a distinct source; Fault Label must never be a sensor")
    sensors = frame.rename(columns={source: target for target, source in zip(NEV_FEATURES, selected)})
    missing = [f for f in NEV_FEATURES if f not in sensors]
    report = {**source, "rows": len(frame), "columns": len(frame.columns),
        "recognized_features": [f for f in NEV_FEATURES if f not in missing],
        "missing_features": missing, "unrecognized_columns": [c for c in trimmed if c not in selected and c != NEV_TARGET],
        "target_column": NEV_TARGET if NEV_TARGET in frame else None,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_sensor_rows": int(frame[selected].duplicated().sum()) if not missing else None,
        "column_types": {name: str(frame[name].dtype) for name in frame},
        "missing_cells": {name: int((frame[name].isna() | frame[name].astype(str).str.strip().eq("")).sum()) for name in frame},
        "mapping": mapping, "compatible": False, "issues": [], "feature_statistics": {},
        "warnings": ["Ranges cannot establish units or preprocessing compatibility. Confirm the original NEV normalization; no scaling or unit conversion is performed.",
                     "No timestamp or vehicle grouping is inferred. Upload scores do not establish unseen-data, temporal or cross-vehicle performance."]}
    if NEV_TARGET in frame:
        target = pd.to_numeric(frame[NEV_TARGET], errors="coerce")
        report["target_check"] = {"invalid_labels": int((~target.isin(NEV_CLASSES)).sum()),
            "documented_class_counts": {str(code): int((target == code).sum()) for code in NEV_CLASSES},
            "label_meaning_verified": False}
    for feature, source_name in zip(NEV_FEATURES, selected):
        if source_name not in frame:
            continue
        series = frame[source_name]
        numeric = pd.to_numeric(series, errors="coerce")
        missing_cells = series.isna() | series.astype(str).str.strip().eq("")
        kinds = np.where(missing_cells, "missing", np.where(numeric.isna(), "nonnumeric",
            np.where(~np.isfinite(numeric), "nonfinite", np.where((numeric < -1e-12) | (numeric > 1+1e-12), "out_of_range", "valid"))))
        for index, kind in enumerate(kinds):
            if kind != "valid" and len(report["issues"]) < 20:
                report["issues"].append({"observation": index+1, "column": source_name, "kind": str(kind),
                    "csv_line": frame.attrs.get("csv_lines", [None]*len(frame))[index]})
        finite = numeric[np.isfinite(numeric)]
        report["feature_statistics"][feature] = {"dtype": str(series.dtype),
            "missing": int(missing_cells.sum()), "invalid": int((kinds != "valid").sum()),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None}
    report["compatible"] = bool(len(frame) and not missing and not any(s["invalid"] for s in report["feature_statistics"].values()))
    report["reason"] = ("Schema/ranges accepted; normalization and context still require confirmation" if report["compatible"]
        else "Missing sensors, invalid cells or empty data; no predictions generated")
    return report, sensors


def process_dataset(content, pipeline, model_version, *, table="", mapping="{}",
                    mode="inspect", normalized=False, labels_confirmed=False):
    report, frame = inspect_dataset(content, table, mapping)
    if mode == "inspect":
        labels = None
        if frame is not None and labels_confirmed and NEV_TARGET in frame:
            possible = pd.to_numeric(frame[NEV_TARGET], errors="coerce")
            if possible.isin(NEV_CLASSES).all():
                labels = possible.astype(int)
        return {"inspection": report, "analytics": build_dashboard_data(frame, labels=labels,
            label_basis="user_confirmed_labels" if labels is not None else "unlabeled",
            normalized=normalized, reference=load_dashboard_reference()) if frame is not None else None}
    if not report["compatible"]:
        details = [report["reason"]]
        if report.get("missing_features"):
            details.append("Missing sensors: " + ", ".join(report["missing_features"]))
        details.extend(f"Observation {item['observation']}, {item['column']}: {item['kind']}"
                       for item in report.get("issues", []))
        raise TelemetryValidationError(". ".join(details))
    if not normalized:
        raise TelemetryValidationError("Explicitly confirm compatible NEV normalization before prediction; [0,1] alone is insufficient")
    if mode == "evaluate" and (not labels_confirmed or NEV_TARGET not in frame):
        raise TelemetryValidationError("Evaluation requires Fault Label and explicit confirmation of its documented 0–3 ground-truth meaning")
    if mode == "evaluate":
        # Recognize exact original training/full files without relying on names.
        # Reformatted files/subsets cannot generally be certified independent.
        metadata = json.loads((Path(__file__).resolve().parents[2] / "models/experiment_2/metadata.json").read_text())
        fingerprint = hashlib.sha256(content).hexdigest()
        if fingerprint in (metadata["raw_sha256"]["training"], metadata["raw_sha256"]["full"]):
            raise TelemetryValidationError("This is a known training-containing dataset. Use prediction/inspection, not independent evaluation")
        target = pd.to_numeric(frame[NEV_TARGET], errors="coerce")
        invalid = target.isna() | ~target.isin(NEV_CLASSES)
        if invalid.any():
            raise TelemetryValidationError(f"Observation {int(np.flatnonzero(invalid)[0])+1}, Fault Label: use integer codes 0, 1, 2, 3; no rows evaluated")
    # Shared validation and complete inference pipeline; only the seven sensors.
    sensors = parse_telemetry_csv(frame.loc[:, list(NEV_FEATURES)].to_csv(index=False).encode())
    output = predict_nev_faults(pipeline, sensors)
    summary = summarize_predictions(output, sensors, model_version)
    result = {"inspection": report, "prediction": summary.model_dump(), "evaluation": None}
    result["analytics"] = summary.analytics
    if mode == "evaluate":
        scores = output[[f"probability_{c}" for c in NEV_CLASSES]].to_numpy()
        metrics = multiclass_classification_metrics(target.astype(int), output.fault_code, scores, NEV_CLASSES)
        metrics.update(multiclass_probability_diagnostics(target.astype(int), scores, list(NEV_CLASSES)))
        result["evaluation"] = {**metrics, "rows_evaluated": len(target), "rows_rejected": 0,
            "class_order": NEV_CLASSES, "independent_test_performance": False,
            "warning": "User-attested labels, not independently verified. Training overlap and dataset provenance cannot generally be established. These are uploaded-data scores, not new held-out test results or proof of calibration."}
        result["analytics"] = build_dashboard_data(sensors, labels=target.astype(int),
            label_basis="user_confirmed_labels", normalized=normalized, reference=load_dashboard_reference())
    # Do not duplicate the large analytics payload inside the prediction summary.
    result["prediction"].pop("analytics", None)
    return result
