"""
Phase 4 output contract: the structured inspection report.

These Pydantic models are what the LLM report generator (ReportGenerator) produces
and what the Phase 5 frontend consumes. They are deliberately provider-agnostic —
nothing here references any specific LLM. `generation_mode`/`generated_by` only ever
distinguish "llm" (validated LLM output) from "fallback" (static template output).

The report is deterministic in STRUCTURE (this schema, Pydantic-validated) but
dynamic in CONTENT (LLM prose grounded in Phase 3 retrieved cases/standards).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

# Content-generation provenance for a single defect report / the run as a whole.
GenerationMode = Literal["llm", "fallback"]

# Coarse severity classification used for the report headline and triage.
# Distinct from CaseResult.severity (a 1-5 int) — see SeverityRationale, which
# carries both the categorical level and the numeric score.
DefectSeverityLevel = Literal["critical", "major", "minor", "observation"]

# Mirror of Phase 3's retrieval_confidence band. Imported rather than redefined so
# the report's confidence vocabulary can never drift from the retrieval pipeline's.
from src.retrieval.schemas import RetrievalConfidence  # noqa: E402


class SeverityRationale(BaseModel):
    """Why a given severity level was assigned to a defect."""

    level: DefectSeverityLevel
    score: int = Field(ge=1, le=5, description="1-5, matching CaseResult.severity")
    rationale: str = Field(description="One sentence explaining the severity assignment")
    ipc_reference: Optional[str] = Field(
        default=None, description="IPC standard section, if a standard was retrieved"
    )


class RootCauseAnalysis(BaseModel):
    """Root-cause conclusion, grounded strictly in retrieved historical cases."""

    primary_cause: str = Field(description="Most likely root cause, grounded in retrieved cases")
    contributing_factors: list[str] = Field(
        default_factory=list, max_length=3, description="Secondary causes, max 3"
    )
    confidence: RetrievalConfidence = Field(description="Mirrors Phase 3 retrieval_confidence")
    evidence_basis: list[str] = Field(
        default_factory=list,
        description="case_ids from retrieved_cases that support this analysis",
    )
    unsupported: bool = Field(
        description="True if analysis could not be grounded in retrieved cases "
        "(retrieved_cases empty or top similarity below threshold)"
    )


class CorrectiveAction(BaseModel):
    """Recommended remediation for a single defect."""

    immediate: str = Field(description="What to do now: remove, quarantine, re-inspect")
    process_adjustment: str = Field(description="Manufacturing process change to prevent recurrence")
    re_inspection: str = Field(description="What to check and when")
    ipc_reference: Optional[str] = Field(
        default=None, description="IPC standard section, if a standard was retrieved"
    )


class DefectReport(BaseModel):
    """Complete report for one detected defect."""

    defect_class: str
    location: str = Field(
        description="Human-readable bbox location (e.g. 'top-left', 'centre'), not raw pixels"
    )
    severity: SeverityRationale
    root_cause: RootCauseAnalysis
    corrective_action: CorrectiveAction
    narrative: str = Field(
        description="2-3 sentence plain-English summary for a factory-floor technician"
    )
    generated_by: GenerationMode = Field(description="'llm' (validated) or 'fallback' (template)")


class GenerationMetadata(BaseModel):
    """Bookkeeping describing how the report as a whole was produced."""

    model_used: Optional[str] = Field(default=None, description="LLM model string; None if all fallback")
    generation_mode: GenerationMode = Field(description="'llm' if any defect used the LLM, else 'fallback'")
    prompt_tokens_used: Optional[int] = Field(
        default=None, description="Total prompt tokens across LLM calls; None if no LLM call succeeded"
    )
    total_defects_processed: int = Field(ge=0)
    fallback_count: int = Field(ge=0, description="Defects that used a fallback template instead of the LLM")


class InspectionReport(BaseModel):
    """Top-level Phase 4 output for one analyzed image. Consumed by the Phase 5 frontend."""

    report_id: str = Field(default_factory=lambda: str(uuid4()), description="UUID minted at creation")
    generated_at: datetime = Field(default_factory=datetime.now)
    total_defects: int = Field(ge=0)
    requires_human_review: bool = Field(
        description="True if ANY input defect had retrieval_metadata.flagged_for_human_review"
    )
    overall_severity: DefectSeverityLevel = Field(description="Highest severity level across all defects")
    defect_reports: list[DefectReport]
    generation_metadata: GenerationMetadata
