"""Localhost MVP: saved-model diagnosis, CSV telemetry and static dashboard.

No model fitting, physical-unit conversion, RUL predictions or risk scoring.
Start from the project root with: python -m uvicorn src.api.app:app --port 8010
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from starlette.concurrency import run_in_threadpool

from src.api.schemas import FaultPrediction, FaultRequest, INPUT_MODE, TelemetrySummary
from src.api.telemetry import (
    MAX_UPLOAD_BYTES, TelemetryLimitError, TelemetryValidationError,
    analyze_telemetry, make_prediction,
)
from src.data.nev_fault import NEV_FEATURES, file_sha256
from src.prediction.nev_fault import predict_nev_faults
from src.storage.model_registry import NEV_PIPELINE_PATH, load_nev_fault_pipeline

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def health_check() -> dict[str, str]:
    """Keep the original lightweight health helper backward compatible."""
    return {"status": "ok", "message": "EV Predictive Maintenance Platform API"}


def create_app(model_path: str | Path = NEV_PIPELINE_PATH) -> FastAPI:
    """Load one trusted local pipeline at startup, never per request or via upload.

    Missing/corrupt artifacts leave the dashboard and health endpoint usable,
    but prediction endpoints return 503. Recovery requires installing the
    trusted artifact/dependencies and restarting, not automatic retraining.
    """
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.pipeline = None
        application.state.model_version = None
        application.state.model_status = "unavailable"
        try:
            application.state.pipeline = load_nev_fault_pipeline(model_path)
            application.state.model_version = "nev-fault-" + file_sha256(model_path)[:12]
            application.state.model_status = "ready"
        except Exception:
            # Details remain in local server logs, not public response bodies.
            LOGGER.exception("Fault model could not be loaded; no retraining attempted")
        yield
        application.state.pipeline = None

    application = FastAPI(
        title="EV Predictive Maintenance Platform", version="0.1.0",
        description="Local proof-of-concept fault diagnosis with normalized inputs; RUL/risk disabled.",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        # Do not echo sensor inputs or nonfinite values into JSON errors.
        errors = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                  for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def require_model():
        pipeline = getattr(application.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Fault model unavailable. Provide the trusted saved pipeline and compatible dependencies, then restart. No automatic training is performed.")
        return pipeline

    @application.get("/health")
    def health():
        ready = getattr(application.state, "model_status", "unavailable") == "ready"
        return {
            **health_check(), "status": "ok" if ready else "degraded",
            "application_status": "running", "fault_model_status": "ready" if ready else "unavailable",
            "model_version": getattr(application.state, "model_version", None),
            "input_mode": INPUT_MODE, "rul_status": "not_connected", "risk_status": "disabled",
            "warning": "Normalized proof-of-concept inputs only. Class probabilities are not real-world failure probabilities. No temporal or cross-vehicle generalization is established.",
        }

    @application.post("/predict-fault", response_model=FaultPrediction)
    def predict_fault(payload: FaultRequest):
        pipeline = require_model()
        sensors = pd.DataFrame([payload.model_dump(by_alias=True)], columns=NEV_FEATURES)
        try:
            output = predict_nev_faults(pipeline, sensors)
            return make_prediction(output.iloc[0], application.state.model_version)
        except Exception:
            LOGGER.exception("Fault inference failed")
            raise HTTPException(status_code=503, detail="Saved model inference failed; check the trusted artifact and compatible dependencies") from None

    @application.post("/analyze-csv", response_model=TelemetrySummary)
    async def analyze_csv(file: UploadFile):
        pipeline = require_model()
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        try:
            return await run_in_threadpool(analyze_telemetry, content, pipeline, application.state.model_version)
        except TelemetryLimitError as error:
            raise HTTPException(status_code=413, detail=str(error)) from None
        except TelemetryValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception:
            LOGGER.exception("CSV inference failed")
            raise HTTPException(status_code=503, detail="Saved model inference failed; no partial results returned") from None

    @application.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
