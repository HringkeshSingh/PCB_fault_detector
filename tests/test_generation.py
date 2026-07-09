"""
Phase 4 report-generation tests.

Run with:  pytest tests/test_generation.py -v

  - Unit tests mock the Ollama client — no real LLM calls, always runnable offline.
  - The integration test (@pytest.mark.integration) requires a running Ollama server
    with the model pulled, and is skipped otherwise.

Run only unit tests:  pytest tests/test_generation.py -v -m "not integration"
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.generation.generator import ReportGenerator
from src.generation.schemas import DefectReport, InspectionReport
from src.retrieval.schemas import (
    AnalysisResult,
    CaseResult,
    RetrievalMetadata,
    RetrievalResult,
    StandardResult,
)
from src.vision.constants import CLASS_NAMES
from src.vision.detect import Detection

# ===========================================================================
# Fixtures & helpers
# ===========================================================================

# DeepPCB / VOC images are 640 or 600; use 600 so grid thirds land on 200/400.
IMG = 600

VALID_LLM_JSON = """
{
  "defect_class": "open",
  "location": "top-left",
  "severity": {"level": "major", "score": 4, "rationale": "Open trace breaks the net.", "ipc_reference": null},
  "root_cause": {
    "primary_cause": "Under-etch severed the trace at the panel edge.",
    "contributing_factors": ["photoresist lift-off"],
    "confidence": "high",
    "evidence_basis": ["PCB-CASE-0001"],
    "unsupported": false
  },
  "corrective_action": {"immediate": "Quarantine board.", "process_adjustment": "Tighten etch control.", "re_inspection": "AOI + continuity.", "ipc_reference": null},
  "narrative": "An open was detected top-left. Historical cases point to under-etch. Quarantine and re-test."
}
"""


def make_case(case_id: str = "PCB-CASE-0001", defect_class: str = "open", sim: float = 0.85) -> CaseResult:
    """Build a realistic CaseResult for prompts/grounding."""
    return CaseResult(
        case_id=case_id,
        defect_class=defect_class,
        component_type="signal trace",
        root_cause="Under-etch severed the trace at the panel edge",
        corrective_action="Tighten etch time control",
        severity=4,
        outcome_notes="AOI opens dropped to zero",
        similarity_score=sim,
    )


def make_result(
    defect_class: str = "open",
    confidence: float = 0.9,
    band: str = "high",
    flagged: bool = False,
    n_cases: int = 2,
    bbox: list[float] | None = None,
) -> RetrievalResult:
    """Build a RetrievalResult with a configurable retrieval band and case count."""
    if bbox is None:
        bbox = [10.0, 10.0, 60.0, 60.0]  # centre ~ (35,35) -> top-left on any size
    cases = [make_case(f"PCB-CASE-000{i}", defect_class) for i in range(n_cases)]
    top_sim = cases[0].similarity_score if cases else 0.0
    return RetrievalResult(
        detection=Detection(defect_class=defect_class, confidence=confidence, bbox=bbox),
        retrieved_cases=cases,
        retrieved_standards=[
            StandardResult(section_id="IPC-A610-S3-01", source_doc="IPC-A-610", excerpt="crit", relevance_score=0.9)
        ],
        retrieval_metadata=RetrievalMetadata(
            retrieval_confidence=band,
            flagged_for_human_review=flagged,
            cases_found=n_cases,
            standards_found=1,
            top_case_similarity=top_sim,
            standards_skipped=False,
        ),
    )


def fallback_generator() -> ReportGenerator:
    """A generator forced into fallback mode (no LLM)."""
    g = ReportGenerator()
    g.llm_available = False
    return g


def llm_generator(chat_return=None, chat_side_effect=None) -> ReportGenerator:
    """A generator with a mocked Ollama client marked available."""
    g = ReportGenerator()
    g.llm_available = True
    g._client = MagicMock()
    if chat_side_effect is not None:
        g._client.chat.side_effect = chat_side_effect
    else:
        g._client.chat.return_value = chat_return
    return g


def chat_response(content: str, prompt_tokens: int = 100, response_tokens: int = 50) -> dict:
    """Shape a mocked Ollama chat response."""
    return {"message": {"content": content}, "prompt_eval_count": prompt_tokens, "eval_count": response_tokens}


# ===========================================================================
# _bbox_to_location
# ===========================================================================


class TestBboxToLocation:
    @pytest.mark.parametrize(
        "center,expected",
        [
            ((100, 100), "top-left"),
            ((300, 100), "top-centre"),
            ((500, 100), "top-right"),
            ((100, 300), "centre-left"),
            ((300, 300), "centre"),
            ((500, 300), "centre-right"),
            ((100, 500), "bottom-left"),
            ((300, 500), "bottom-centre"),
            ((500, 500), "bottom-right"),
        ],
    )
    def test_all_nine_grid_positions(self, center, expected):
        cx, cy = center
        bbox = [cx - 5, cy - 5, cx + 5, cy + 5]
        assert ReportGenerator._bbox_to_location(bbox, IMG, IMG) == expected

    def test_none_bbox_returns_location_unknown(self):
        assert ReportGenerator._bbox_to_location(None, IMG, IMG) == "location unknown"

    def test_malformed_bbox_returns_location_unknown(self):
        assert ReportGenerator._bbox_to_location([1, 2, 3], IMG, IMG) == "location unknown"


# ===========================================================================
# _generate_fallback
# ===========================================================================


class TestFallback:
    @pytest.mark.parametrize("defect_class", CLASS_NAMES)
    def test_fallback_for_every_class(self, defect_class):
        g = fallback_generator()
        report = g.generate_report(AnalysisResult(total_detections=1, results=[make_result(defect_class)]), IMG, IMG)
        dr = report.defect_reports[0]
        assert isinstance(dr, DefectReport)
        assert dr.generated_by == "fallback"
        assert dr.root_cause.unsupported is True
        assert dr.defect_class == defect_class
        assert dr.root_cause.evidence_basis == []

    def test_fallback_location_from_bbox(self):
        g = fallback_generator()
        result = make_result("open", bbox=[560, 560, 590, 590])  # bottom-right on 600px
        report = g.generate_report(AnalysisResult(total_detections=1, results=[result]), IMG, IMG)
        assert report.defect_reports[0].location == "bottom-right"


# ===========================================================================
# generate_report — orchestration
# ===========================================================================


class TestGenerateReport:
    def test_zero_detections(self):
        g = fallback_generator()
        report = g.generate_report(AnalysisResult(total_detections=0, results=[]))
        assert isinstance(report, InspectionReport)
        assert report.total_defects == 0
        assert report.defect_reports == []
        assert report.overall_severity == "observation"
        assert report.generation_metadata.total_defects_processed == 0

    def test_llm_valid_json_marks_generated_by_llm(self):
        g = llm_generator(chat_return=chat_response(VALID_LLM_JSON))
        report = g.generate_report(AnalysisResult(total_detections=1, results=[make_result("open")]), IMG, IMG)
        dr = report.defect_reports[0]
        assert dr.generated_by == "llm"
        assert report.generation_metadata.generation_mode == "llm"
        assert report.generation_metadata.model_used == g.model
        assert report.generation_metadata.prompt_tokens_used == 100

    def test_llm_invalid_json_falls_back(self):
        g = llm_generator(chat_return=chat_response("this is not json"))
        report = g.generate_report(AnalysisResult(total_detections=1, results=[make_result("open")]), IMG, IMG)
        assert report.defect_reports[0].generated_by == "fallback"
        assert report.generation_metadata.fallback_count == 1

    def test_llm_exception_falls_back_without_raising(self):
        g = llm_generator(chat_side_effect=RuntimeError("connection reset"))
        report = g.generate_report(AnalysisResult(total_detections=1, results=[make_result("open")]), IMG, IMG)
        assert report.defect_reports[0].generated_by == "fallback"

    def test_uncertain_band_skips_llm(self):
        # Even with a working LLM, an uncertain retrieval band must use fallback.
        g = llm_generator(chat_return=chat_response(VALID_LLM_JSON))
        result = make_result("open", confidence=0.3, band="uncertain", flagged=True)
        report = g.generate_report(AnalysisResult(total_detections=1, results=[result]), IMG, IMG)
        assert report.defect_reports[0].generated_by == "fallback"
        g._client.chat.assert_not_called()

    def test_requires_human_review_true_if_any_flagged(self):
        g = fallback_generator()
        results = [make_result("open", flagged=False), make_result("spur", flagged=True)]
        report = g.generate_report(AnalysisResult(total_detections=2, results=results), IMG, IMG)
        assert report.requires_human_review is True

    def test_requires_human_review_false_if_none_flagged(self):
        g = fallback_generator()
        results = [make_result("open", flagged=False), make_result("spur", flagged=False)]
        report = g.generate_report(AnalysisResult(total_detections=2, results=results), IMG, IMG)
        assert report.requires_human_review is False

    def test_overall_severity_is_highest(self):
        # open fallback = major, spur fallback = minor -> overall major.
        g = fallback_generator()
        results = [make_result("spur"), make_result("open")]
        report = g.generate_report(AnalysisResult(total_detections=2, results=results), IMG, IMG)
        assert report.overall_severity == "major"

    def test_llm_output_never_raw_contributing_factors_truncated(self):
        # LLM returns 4 contributing factors; generator must normalise to <=3, not error out.
        four = VALID_LLM_JSON.replace('["photoresist lift-off"]', '["a", "b", "c", "d"]')
        g = llm_generator(chat_return=chat_response(four))
        report = g.generate_report(AnalysisResult(total_detections=1, results=[make_result("open")]), IMG, IMG)
        dr = report.defect_reports[0]
        assert dr.generated_by == "llm"
        assert len(dr.root_cause.contributing_factors) == 3


# ===========================================================================
# Integration — requires a live Ollama server with the model pulled
# ===========================================================================

_OLLAMA_LIVE = ReportGenerator().llm_available


@pytest.mark.integration
@pytest.mark.skipif(not _OLLAMA_LIVE, reason="requires a running Ollama server with the model pulled")
class TestOllamaIntegration:
    def test_llm_generates_valid_grounded_report(self):
        g = ReportGenerator()
        result = make_result("open", n_cases=3)
        report = g.generate_report(AnalysisResult(total_detections=1, results=[result]), IMG, IMG)
        dr = report.defect_reports[0]
        # Whatever the mode, the output must validate as a DefectReport (Pydantic already enforced).
        assert isinstance(dr, DefectReport)
        if dr.generated_by == "llm":
            # Grounding check: evidence_basis must reference only retrieved case_ids.
            retrieved_ids = {c.case_id for c in result.retrieved_cases}
            assert set(dr.root_cause.evidence_basis).issubset(retrieved_ids)
