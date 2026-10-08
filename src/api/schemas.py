"""Strict API contracts for normalized, proof-of-concept fault diagnosis."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.nev_fault import NEV_FEATURES, NORMALIZED_RANGE_TOLERANCE

INPUT_MODE = "normalized_proof_of_concept"
NormalizedValue = Annotated[
    float,
    Field(strict=True, ge=-NORMALIZED_RANGE_TOLERANCE,
          le=1 + NORMALIZED_RANGE_TOLERANCE, allow_inf_nan=False),
]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class FaultRequest(BaseModel):
    """Exact original column aliases, with no string/bool numeric coercion."""

    model_config = ConfigDict(extra="forbid", validate_by_alias=True, validate_by_name=False)
    voltage: NormalizedValue = Field(alias=NEV_FEATURES[0])
    current: NormalizedValue = Field(alias=NEV_FEATURES[1])
    motor_speed: NormalizedValue = Field(alias=NEV_FEATURES[2])
    temperature: NormalizedValue = Field(alias=NEV_FEATURES[3])
    vibration: NormalizedValue = Field(alias=NEV_FEATURES[4])
    ambient_temperature: NormalizedValue = Field(alias=NEV_FEATURES[5])
    humidity: NormalizedValue = Field(alias=NEV_FEATURES[6])

    @field_validator("*", mode="before")
    @classmethod
    def require_json_number(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Use a JSON number, not a string, boolean or null")
        return value


class FaultPrediction(BaseModel):
    predicted_class: Literal[0, 1, 2, 3]
    predicted_fault: Literal["Normal", "Motor Fault", "Inverter Fault", "Battery Fault"]
    model_confidence: Probability
    class_probabilities: dict[str, Probability]
    status: Literal["normal", "fault_detected"]
    input_mode: Literal["normalized_proof_of_concept"] = INPUT_MODE
    model_version: str


class ClassSummary(BaseModel):
    count: int
    percentage: float


class TelemetrySummary(BaseModel):
    rows_analyzed: int
    classes: dict[str, ClassSummary]
    dominant_abnormal_fault: str | None
    dominant_abnormal_faults: list[str]
    mean_model_confidence: Probability
    input_mode: Literal["normalized_proof_of_concept"] = INPUT_MODE
    model_version: str
    temporal_analysis: Literal[False] = False
    row_order: Literal["uploaded_file_order_not_time"] = "uploaded_file_order_not_time"
    predictions_returned: int
    predictions_truncated: bool
    predictions: list[FaultPrediction]
