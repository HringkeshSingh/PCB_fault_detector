"""Pydantic schemas for the detection API."""

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    defect_class: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")


class DetectResponse(BaseModel):
    detections: list[DetectionItem]
    image_width: int
    image_height: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ReportHealthResponse(BaseModel):
    """Phase 4 report-generator status, so the frontend can show a fallback-mode indicator."""

    llm_available: bool
    model: str | None = Field(default=None, description="LLM model string when available, else None")
    fallback_mode: bool = Field(description="True when reports will use static templates, not the LLM")
