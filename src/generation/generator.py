"""
Phase 4 report generator.

ReportGenerator turns a Phase 3 AnalysisResult into a Pydantic-validated
InspectionReport. It uses a LOCAL Ollama model (no API key, offline); if Ollama is
unreachable, the model is not pulled, a call fails, or the model returns output that
fails Pydantic validation, it degrades to static fallback templates.

Hard guarantees:
- generate_report() NEVER raises — errors go to logs, a valid InspectionReport always returns.
- Raw LLM text is NEVER surfaced: every LLM response is JSON-parsed and Pydantic-validated
  before use; anything else falls back.
- Fallback output is schema-identical to LLM output; only generated_by="fallback" and
  root_cause.unsupported=True distinguish it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from src.generation.prompts import (
    FALLBACK_TEMPLATES,
    SYSTEM_PROMPT,
    bbox_to_location,
    build_defect_report_prompt,
)
from src.generation.schemas import (
    CorrectiveAction,
    DefectReport,
    DefectSeverityLevel,
    GenerationMetadata,
    InspectionReport,
    RootCauseAnalysis,
    SeverityRationale,
)
from src.retrieval.schemas import AnalysisResult, RetrievalResult
from src.vision.constants import IMAGE_HEIGHT, IMAGE_WIDTH

logger = logging.getLogger(__name__)

# Highest-first ordering for computing the report's overall severity.
_SEVERITY_ORDER: list[DefectSeverityLevel] = ["critical", "major", "minor", "observation"]
_SEVERITY_RANK: dict[str, int] = {level: i for i, level in enumerate(_SEVERITY_ORDER)}

DEFAULT_MODEL = "llama3.2"


class ReportGenerator:
    """Generates a validated InspectionReport from a Phase 3 AnalysisResult, LLM or fallback."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> None:
        """Build the generator and probe local Ollama availability (never raises)."""
        # OLLAMA_MODEL / OLLAMA_HOST env vars override the defaults if set.
        self.model = os.environ.get("OLLAMA_MODEL", model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._host = os.environ.get("OLLAMA_HOST") or None
        self._client: Any = None
        self.llm_available = False
        self._probe_ollama()

    def _probe_ollama(self) -> None:
        """Set self.llm_available by checking the Ollama server is up and the model is pulled."""
        try:
            from ollama import Client
        except ImportError:
            logger.warning("ollama package not installed — /report will use fallback templates")
            return
        try:
            self._client = Client(host=self._host)
            listed = self._client.list()
            names = {getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                     for m in getattr(listed, "models", None) or listed.get("models", [])}
            names.discard(None)
            if self._model_present(names):
                self.llm_available = True
                logger.info("Ollama available; using model '%s'", self.model)
            else:
                logger.warning(
                    "Ollama reachable but model '%s' not pulled (have: %s) — "
                    "run `ollama pull %s`; using fallback templates until then",
                    self.model, sorted(names), self.model,
                )
        except Exception as exc:
            logger.warning("Ollama not reachable (%s) — /report will use fallback templates", exc)

    def _model_present(self, names: set[str]) -> bool:
        """Return True if self.model matches a pulled model name (tolerating the ':latest' tag)."""
        if self.model in names or f"{self.model}:latest" in names:
            return True
        return any(str(n).split(":", 1)[0] == self.model for n in names)

    @staticmethod
    def _bbox_to_location(
        bbox: list[float] | None,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> str:
        """Delegate to the shared bbox->location translation used by the prompt builder."""
        return bbox_to_location(bbox, image_width, image_height)

    def generate_report(
        self,
        analysis: AnalysisResult,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> InspectionReport:
        """Produce a validated InspectionReport for one analyzed image; never raises."""
        defect_reports: list[DefectReport] = []
        prompt_tokens_total = 0
        fallback_count = 0
        any_llm = False

        for result in analysis.results:
            use_llm = self.llm_available and result.retrieval_metadata.retrieval_confidence != "uncertain"
            if use_llm:
                report, tokens = self._generate_with_llm(result, image_width, image_height)
                if report.generated_by == "llm":
                    any_llm = True
                    prompt_tokens_total += tokens or 0
                else:
                    fallback_count += 1
            else:
                report = self._generate_fallback(result, image_width, image_height)
                fallback_count += 1
            defect_reports.append(report)

        requires_human_review = any(
            r.retrieval_metadata.flagged_for_human_review for r in analysis.results
        )
        overall_severity = self._overall_severity(defect_reports)
        metadata = GenerationMetadata(
            model_used=self.model if any_llm else None,
            generation_mode="llm" if any_llm else "fallback",
            prompt_tokens_used=prompt_tokens_total if any_llm else None,
            total_defects_processed=len(analysis.results),
            fallback_count=fallback_count,
        )
        return InspectionReport(
            total_defects=len(defect_reports),
            requires_human_review=requires_human_review,
            overall_severity=overall_severity,
            defect_reports=defect_reports,
            generation_metadata=metadata,
        )

    @staticmethod
    def _overall_severity(reports: list[DefectReport]) -> DefectSeverityLevel:
        """Return the highest severity level across reports (observation if there are none)."""
        if not reports:
            return "observation"
        return min(reports, key=lambda r: _SEVERITY_RANK[r.severity.level]).severity.level

    def _generate_with_llm(
        self,
        result: RetrievalResult,
        image_width: int,
        image_height: int,
    ) -> tuple[DefectReport, Optional[int]]:
        """Call Ollama for one defect; validate the JSON, or fall back. Returns (report, prompt_tokens)."""
        det = result.detection
        location = self._bbox_to_location(det.bbox, image_width, image_height)
        band = result.retrieval_metadata.retrieval_confidence
        user_prompt = build_defect_report_prompt(
            defect_class=det.defect_class,
            confidence_band=band,
            flagged_for_human_review=result.retrieval_metadata.flagged_for_human_review,
            location=location,
            cases=result.retrieved_cases,
            standards=result.retrieved_standards,
        )
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
            )
        except Exception as exc:
            logger.error(
                "LLM call failed | model=%s defect_class=%s error=%s — falling back",
                self.model, det.defect_class, exc,
            )
            return self._generate_fallback(result, image_width, image_height), None

        raw, prompt_tokens, response_tokens = self._extract_response(response)
        try:
            data = self._parse_json(raw)
            # Force system-controlled fields; never trust the model to echo them.
            data["defect_class"] = det.defect_class
            data["location"] = location
            data["generated_by"] = "llm"
            if isinstance(data.get("root_cause"), dict):
                factors = data["root_cause"].get("contributing_factors")
                if isinstance(factors, list):
                    data["root_cause"]["contributing_factors"] = factors[:3]
            report = DefectReport.model_validate(data)
        except Exception as exc:
            logger.warning("LLM raw response | defect_class=%s: %s", det.defect_class, raw)
            logger.warning(
                "LLM output validation failed | model=%s defect_class=%s error=%s — falling back",
                self.model, det.defect_class, exc,
            )
            return self._generate_fallback(result, image_width, image_height), None

        logger.info(
            "LLM call | model=%s defect_class=%s prompt_tokens=%s response_tokens=%s generation_mode=llm",
            self.model, det.defect_class, prompt_tokens, response_tokens,
        )
        return report, prompt_tokens

    @staticmethod
    def _extract_response(response: Any) -> tuple[str, Optional[int], Optional[int]]:
        """Pull (content, prompt_tokens, response_tokens) from an Ollama chat response."""
        def _get(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        message = _get(response, "message")
        content = _get(message, "content") or ""
        return content, _get(response, "prompt_eval_count"), _get(response, "eval_count")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse an LLM JSON response, tolerating stray markdown code fences."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip().rstrip("`").strip()
        return json.loads(text)

    def _generate_fallback(
        self,
        result: RetrievalResult,
        image_width: int,
        image_height: int,
    ) -> DefectReport:
        """Build a schema-valid DefectReport from static templates; generated_by='fallback'."""
        det = result.detection
        location = self._bbox_to_location(det.bbox, image_width, image_height)
        template = FALLBACK_TEMPLATES.get(det.defect_class, self._generic_fallback(det.defect_class))
        logger.info(
            "Fallback report | defect_class=%s generation_mode=fallback location=%s",
            det.defect_class, location,
        )
        return DefectReport(
            defect_class=det.defect_class,
            location=location,
            severity=SeverityRationale(
                level=template["severity_level"],
                score=template["severity_score"],
                rationale=template["severity_rationale"],
                ipc_reference=None,
            ),
            root_cause=RootCauseAnalysis(
                primary_cause=template["primary_cause"],
                contributing_factors=list(template["contributing_factors"])[:3],
                confidence=result.retrieval_metadata.retrieval_confidence,
                evidence_basis=[],
                unsupported=True,
            ),
            corrective_action=CorrectiveAction(
                immediate=template["immediate"],
                process_adjustment=template["process_adjustment"],
                re_inspection=template["re_inspection"],
                ipc_reference=None,
            ),
            narrative=template["narrative"].format(defect_class=det.defect_class, location=location),
            generated_by="fallback",
        )

    @staticmethod
    def _generic_fallback(defect_class: str) -> dict[str, Any]:
        """Last-resort template for an unknown defect class (keeps the contract intact)."""
        return {
            "severity_level": "observation",
            "severity_score": 3,
            "severity_rationale": f"Severity for '{defect_class}' could not be assessed from available data.",
            "primary_cause": f"A '{defect_class}' defect was detected; root-cause analysis is not supported by historical data.",
            "contributing_factors": [],
            "immediate": "Quarantine the board and route to a human inspector.",
            "process_adjustment": "No process adjustment can be recommended without a matched historical case.",
            "re_inspection": "Manual inspection by a qualified technician.",
            "narrative": "A {defect_class} defect was detected at the {location} of the board, but no historical data was available to analyze it. Route the board to a human inspector.",
        }
