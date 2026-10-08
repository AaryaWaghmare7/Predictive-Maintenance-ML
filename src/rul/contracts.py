"""Contract for a separately approved degradation/run-to-failure model."""

from typing import Annotated, Literal, Protocol

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class RULPrediction(BaseModel):
    """Require an actual model result: estimated_rul has no numerical default.

    Unit and model version must come from that component's validated model.
    Observation time is optional for cycle-only datasets; never invent a date.
    """

    model_config = ConfigDict(extra="forbid")
    component: Annotated[str, Field(min_length=1)]
    machine_id: Annotated[str, Field(min_length=1)]
    estimated_rul: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    unit: Literal["operating_hours", "cycles", "seconds"]
    model_version: Annotated[str, Field(min_length=1)]
    observation_time: AwareDatetime | None = None


class RULModel(Protocol):
    """Future component adapter; each model owns its schema/preprocessing.

    History must belong to one machine and include past observations only.
    No implementation or model instance is registered in the current MVP.
    """

    feature_names: tuple[str, ...]
    model_version: str

    def predict(self, history: pd.DataFrame) -> RULPrediction:
        """Estimate remaining life in the model's documented unit."""
        ...
