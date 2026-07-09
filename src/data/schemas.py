"""JSON schema for historical PCB defect cases (Phase 2)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

DefectClass = Literal["open", "short", "mousebite", "spur", "copper", "pin-hole"]

DEFECT_CLASSES: list[DefectClass] = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]


class HistoricalDefectCase(BaseModel):
    """
    A resolved historical PCB inspection case stored for RAG retrieval (Phase 3).

    JSON Schema (informal):
      - case_id: unique identifier (e.g. PCB-CASE-0042)
      - defect_class: one of the 6 DeepPCB defect types
      - component_type: affected PCB element (trace, pad, via, etc.)
      - root_cause: manufacturing/process root cause
      - corrective_action: remediation taken on the line
      - severity: 1 (minor) to 5 (critical)
      - date_recorded: ISO date when case was logged
      - outcome_notes: post-fix verification notes
    """

    case_id: str = Field(description="Unique case identifier")
    defect_class: DefectClass
    component_type: str = Field(description="Affected PCB feature or component area")
    root_cause: str
    corrective_action: str
    severity: int = Field(ge=1, le=5, description="1=minor, 5=critical")
    date_recorded: date
    outcome_notes: str


class HistoricalCaseCollection(BaseModel):
    """Wrapper for serialized case list."""

    cases: list[HistoricalDefectCase]
    version: str = "1.0"
    description: str = "Synthetic historical PCB defect cases for RAG ingestion"

    def to_embedding_text(self, case: HistoricalDefectCase) -> str:
        """Combined text used for semantic embedding."""
        return (
            f"Defect: {case.defect_class}. "
            f"Component: {case.component_type}. "
            f"Root cause: {case.root_cause}. "
            f"Corrective action: {case.corrective_action}. "
            f"Notes: {case.outcome_notes}"
        )
