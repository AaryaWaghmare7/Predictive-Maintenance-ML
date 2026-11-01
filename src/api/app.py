"""Localhost MVP: saved-model diagnosis, CSV telemetry and static dashboard.

No model fitting, physical-unit conversion, RUL predictions or risk scoring.
Start from the project root with: python -m uvicorn src.api.app:app --port 8012
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from typing import Literal
import base64
import hashlib
import re
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import pandas as pd
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.schemas import FaultPrediction, FaultRequest, INPUT_MODE, TelemetrySummary
from src.api.telemetry import (
    MAX_UPLOAD_BYTES, TelemetryLimitError, TelemetryValidationError,
    analyze_telemetry, make_prediction,
)
from src.data.nev_fault import NEV_FEATURES, file_sha256
from src.prediction.nev_fault import predict_nev_faults
from src.storage.model_registry import NEV_PIPELINE_PATH, load_nev_fault_pipeline
from src.api.datasets import process_dataset
from src.api.security import AccessPolicy
from src.prediction.guidance import fault_guidance
from src.eda.dashboard import load_dashboard_reference

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BYTES = 6 * 1024 * 1024  # Multipart envelope; CSV itself remains 5 MiB.


class RequestSafetyMiddleware:
    """Bound actual request bytes before multipart parsing, including chunked uploads.

    This is a per-request guard, not authentication, rate limiting or a substitute
    for a hosting provider's ingress controls. No body is copied or persisted here.
    """

    def __init__(self, app: ASGIApp, policy):
        self.app = app
        self.policy = policy
        inline = re.search(r'<script id="local-preview-guard">([\s\S]*?)</script>', (STATIC_DIR / "index.html").read_text()).group(1)
        self.script_hash = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
        docs = get_swagger_ui_html(openapi_url="/openapi.json", title="EVision - Swagger UI",
                                   oauth2_redirect_url="/docs/oauth2-redirect").body.decode()
        self.docs_hash = base64.b64encode(hashlib.sha256(re.findall(r"<script>([\s\S]*?)</script>", docs)[0].encode()).digest()).decode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        upload = scope["path"] in ("/analyze-csv", "/analyze-dataset") and scope["method"] == "POST"
        limit = MAX_REQUEST_BYTES if upload else 16 * 1024

        async def safe_send(message):
            if message["type"] == "http.response.start":
                # Local-only Swagger uses one CDN and an exact inline-script hash.
                docs_csp = ("default-src 'self'; script-src https://cdn.jsdelivr.net 'sha256-" + self.docs_hash + "'; style-src https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
                app_csp = ("default-src 'self'; script-src 'self' 'sha256-" + self.script_hash + "'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
                message["headers"] = list(message["headers"]) + [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", (docs_csp if scope["path"] == "/docs" and not self.policy.production else app_csp).encode()),
                ]
                if self.policy.production and scope.get("scheme") == "https":
                    message["headers"].append((b"strict-transport-security", b"max-age=31536000"))
            await send(message)

        rejected = self.policy.check(scope)
        if rejected:
            return await self.policy.rejection(*rejected)(scope, receive, safe_send)
        if upload and self.policy.active_uploads >= 2:
            return await self.policy.rejection(429, "Two uploads are already processing; retry later")(scope, receive, safe_send)
        length = dict(scope["headers"]).get(b"content-length", b"0")
        try:
            size = int(length)
            if size < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, safe_send)
        if size > limit:
            return await JSONResponse({"detail": "Request exceeds the allowed body size"}, status_code=413)(scope, receive, safe_send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail="Request exceeds the allowed body size")
            return message

        if upload:
            self.policy.active_uploads += 1
        try:
            await self.app(scope, limited_receive, safe_send)
        finally:
            if upload:
                self.policy.active_uploads -= 1


def health_check() -> dict[str, str]:
    """Keep the original lightweight health helper backward compatible."""
    return {"status": "ok", "message": "EVision API — AI-Powered EV Predictive Maintenance"}


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

    policy = AccessPolicy()
    application = FastAPI(
        title="EVision", version="0.1.0",
        description="Local proof-of-concept fault diagnosis with normalized inputs; RUL/risk disabled.",
        lifespan=lifespan,
        docs_url=None if policy.production else "/docs",
        redoc_url=None, openapi_url=None if policy.production else "/openapi.json",
    )
    application.state.access_policy = policy
    application.add_middleware(RequestSafetyMiddleware, policy=policy)

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
            "authentication_required": bool(policy.token),
            "input_mode": INPUT_MODE, "rul_status": "not_connected", "risk_status": "disabled",
            "warning": "Normalized proof-of-concept inputs only. Class probabilities are not real-world failure probabilities. No temporal or cross-vehicle generalization is established.",
        }

    @application.post("/predict-fault", response_model=FaultPrediction)
    def predict_fault(payload: FaultRequest):
        pipeline = require_model()
        sensors = pd.DataFrame([payload.model_dump(by_alias=True)], columns=NEV_FEATURES)
        try:
            output = predict_nev_faults(pipeline, sensors)
            prediction = make_prediction(output.iloc[0], application.state.model_version)
            prediction.guidance = fault_guidance(prediction.predicted_class, sensors.iloc[0])
            return prediction
        except Exception:
            LOGGER.exception("Fault inference failed")
            raise HTTPException(status_code=503, detail="Saved model inference failed; check the trusted artifact and compatible dependencies") from None

    @application.get("/ready", include_in_schema=False)
    def readiness():
        """Model-aware health check for a future service; /health stays compatible."""
        current = health()
        return JSONResponse(current, status_code=200 if current["fault_model_status"] == "ready" else 503)

    @application.post("/analyze-csv", response_model=TelemetrySummary)
    async def analyze_csv(file: UploadFile = File(...)):
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

    @application.get("/analytics/reference")
    def analytics_reference():
        reference = load_dashboard_reference()
        if reference is None:
            raise HTTPException(503, "Verified analytics reference is unavailable; no training values are invented")
        return reference

    @application.post("/analyze-dataset")
    async def dataset(file: UploadFile = File(...), mode: Literal["inspect", "predict", "evaluate"] = Form("inspect"),
                      table: str = Form("", max_length=128), mapping: str = Form("{}", max_length=2048),
                      normalized: bool = Form(False), labels_confirmed: bool = Form(False)):
        pipeline = require_model() if mode != "inspect" else None
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        try:
            return await run_in_threadpool(process_dataset, content, pipeline,
                application.state.model_version, table=table, mapping=mapping,
                mode=mode, normalized=normalized, labels_confirmed=labels_confirmed)
        except TelemetryLimitError as error:
            raise HTTPException(413, str(error)) from None
        except TelemetryValidationError as error:
            raise HTTPException(422, str(error)) from None
        except Exception:
            LOGGER.exception("Dataset processing failed")
            raise HTTPException(503, "Dataset processing failed; no partial results returned") from None

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
