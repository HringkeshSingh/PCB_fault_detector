"""
Phase 3 confidence scoring.

Computes the final retrieval_confidence rating and flagged_for_human_review
flag for a RetrievalResult already assembled by PCBRetriever.combine().
"""

from __future__ import annotations

from src.retrieval.schemas import RetrievalConfidence, RetrievalResult

# Detection confidence below this makes retrieval_confidence "uncertain"
# regardless of similarity — distinct from query_builder.py's 0.20 cutoff,
# which only governs RetrievalQuery.confidence_band/is_uncertain.
UNCERTAIN_DETECTION_THRESHOLD: float = 0.40
HIGH_SIMILARITY_THRESHOLD: float = 0.80
MEDIUM_SIMILARITY_THRESHOLD: float = 0.65
HUMAN_REVIEW_SIMILARITY_THRESHOLD: float = 0.55
HUMAN_REVIEW_MIN_CASES: int = 2
HIGH_DETECTION_THRESHOLD: float = 0.60


def _score_retrieval_confidence(top_sim: float, det_conf: float) -> RetrievalConfidence:
    """Assign the retrieval_confidence band from top case similarity and detection confidence."""
    if det_conf < UNCERTAIN_DETECTION_THRESHOLD:
        return "uncertain"
    if top_sim > HIGH_SIMILARITY_THRESHOLD and det_conf > HIGH_DETECTION_THRESHOLD:
        return "high"
    if top_sim > MEDIUM_SIMILARITY_THRESHOLD or det_conf > HIGH_DETECTION_THRESHOLD:
        return "medium"
    return "low"


def _should_flag_for_human_review(top_sim: float, det_conf: float, cases_found: int) -> bool:
    """Return True if any human-review trigger condition is met."""
    if det_conf < UNCERTAIN_DETECTION_THRESHOLD:
        return True
    if cases_found < HUMAN_REVIEW_MIN_CASES:
        return True
    if top_sim < HUMAN_REVIEW_SIMILARITY_THRESHOLD:
        return True
    return False


class ConfidenceScorer:
    """Computes retrieval_confidence and flagged_for_human_review for a RetrievalResult."""

    def score(self, result: RetrievalResult) -> RetrievalResult:
        """Mutate result.retrieval_metadata in place with the final confidence rating and flag."""
        cases = result.retrieved_cases
        top_sim = cases[0].similarity_score if cases else 0.0
        det_conf = result.detection.confidence

        result.retrieval_metadata.top_case_similarity = top_sim
        result.retrieval_metadata.retrieval_confidence = _score_retrieval_confidence(top_sim, det_conf)
        result.retrieval_metadata.flagged_for_human_review = _should_flag_for_human_review(
            top_sim, det_conf, len(cases)
        )
        return result
