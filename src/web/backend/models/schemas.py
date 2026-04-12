"""Pydantic schemas for ViMAD web API request/response."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


ConfigType = Literal[
    "phobert",
    "full_n2k3", "full_n2k5", "full_n3k3", "full_n3k5", "full_n4k3", "full_n4k5",
    "hybrid_n2k3", "hybrid_n2k5", "hybrid_n3k3", "hybrid_n3k5", "hybrid_n4k3", "hybrid_n4k5",
]


class PredictRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim to verify")
    evidence: str = Field(..., min_length=1, description="Gold evidence")
    config: ConfigType = Field(default="phobert", description="Model/debate config")


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
