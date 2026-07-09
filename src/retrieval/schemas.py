"""
Phase 3 output contract.

These Pydantic models are the interface between the RAG retrieval pipeline
(Phase 3) and the LLM report generator (Phase 4). Phase 4 imports
AnalysisResult (and its nested models) directly from this module — field
names and types here must not change without updating Phase 4 in lockstep.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.vision.detect import Detection

ConfidenceBand = Literal["high", "medium", "low", "uncertain"]
RetrievalConfidence = Literal["high", "medium", "low", "uncertain"]


class RetrievalQuery(BaseModel):
    """One semantic query built from a single detected defect class."""

    defect_class: str = Field(description="Must match a ChromaDB defect_class metadata value exactly")
    confidence: float = Field(ge=0.0, le=1.0, description="Raw detection confidence from Phase 1")
    confidence_band: ConfidenceBand
    query_text: str = Field(description="Constructed semantic query string sent to ChromaDB")
    is_uncertain: bool = Field(description="True if confidence < UNCERTAIN threshold")


class CaseResult(BaseModel):
    """One historical case retrieved from the pcb_defect_cases collection."""

    case_id: str
    defect_class: str
    component_type: str = Field(default="", description="Affected PCB feature or component area")
    date_recorded: str = Field(default="", description="ISO date the case was logged, for citation")
    root_cause: str
    corrective_action: str
    severity: int = Field(ge=1, le=5)
    outcome_notes: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class StandardResult(BaseModel):
    """One SOP/standards excerpt retrieved from the pcb_standards collection."""

    section_id: str
    source_doc: str
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class RetrievalMetadata(BaseModel):
    """Quality/bookkeeping fields describing one RetrievalResult."""

    retrieval_confidence: RetrievalConfidence
    flagged_for_human_review: bool
    cases_found: int = Field(ge=0)
    standards_found: int = Field(ge=0)
    top_case_similarity: float = Field(ge=0.0, le=1.0)
    standards_skipped: bool


class RetrievalResult(BaseModel):
    """Complete retrieval output for one detected defect."""

    detection: Detection
    retrieved_cases: list[CaseResult]
    retrieved_standards: list[StandardResult]
    retrieval_metadata: RetrievalMetadata


class AnalysisResult(BaseModel):
    """
    Top-level output of the Phase 3 pipeline for one analyzed image.

    This is the exact object Phase 4 (LLM report generator) consumes.
    """

    total_detections: int = Field(ge=0)
    results: list[RetrievalResult]
