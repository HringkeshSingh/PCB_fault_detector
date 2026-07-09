"""
Phase 3 query builder.

Converts Phase 1 detection dicts into RetrievalQuery objects ready for
ChromaDB semantic search. One query is built per unique defect class.
"""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.schemas import ConfidenceBand, RetrievalQuery

logger = logging.getLogger(__name__)

# Confidence-band boundaries for RetrievalQuery.confidence_band / is_uncertain.
# NOTE: this is a distinct, more permissive threshold than
# confidence.py's UNCERTAIN_DETECTION_THRESHOLD (0.40), which governs the
# separate retrieval_confidence rating computed later by ConfidenceScorer.
HIGH_BAND_MIN: float = 0.70
MEDIUM_BAND_MIN: float = 0.40
LOW_BAND_MIN: float = 0.20  # below this, confidence_band == "uncertain"

# Short semantic descriptions per defect class, built ONLY from vocabulary
# that already appears in data/cases/historical_cases.json root_cause and
# corrective_action fields, so query embeddings land close to the stored
# case embeddings in vector space.
_CLASS_VOCABULARY: dict[str, str] = {
    "open": (
        "open circuit or broken trace from under-etch, incomplete via barrel "
        "plating, mechanical nick during depanelization, or drill misregistration "
        "during lamination"
    ),
    "short": (
        "short circuit from solder bridge, excess paste on stencil, ionic "
        "contamination causing dendritic short, solder mask misalignment, or "
        "copper sliver left after CAM cleanup"
    ),
    "mousebite": (
        "mousebite edge notch from lateral undercut on trace sidewalls, "
        "aggressive etching with etchant ORP drift, etch factor mismatch, or "
        "photoresist edge damage"
    ),
    "spur": (
        "copper spur or foreign copper fragment left in a clearance zone from "
        "photoresist scumming, misaligned artwork extending into a mask opening, "
        "or a partially developed resist island"
    ),
    "copper": (
        "spurious copper nodule or island in a mask-free zone from a photoresist "
        "pinhole, resin smear, copper foil burr from a routing bit, or acid trap "
        "plating buildup"
    ),
    "pin-hole": (
        "pin-hole void or non-coating bubble in copper plating from entrapped "
        "air, an electroless copper seed layer defect, organic residue, or "
        "laser drill debris at the capture pad"
    ),
}


def _confidence_band(confidence: float) -> ConfidenceBand:
    """Map a raw confidence float to its discrete band."""
    if confidence >= HIGH_BAND_MIN:
        return "high"
    if confidence >= MEDIUM_BAND_MIN:
        return "medium"
    if confidence >= LOW_BAND_MIN:
        return "low"
    return "uncertain"


def _build_query_text(defect_class: str) -> str:
    """Build a semantic query string using vocabulary drawn from historical_cases.json."""
    vocabulary = _CLASS_VOCABULARY.get(defect_class, f"{defect_class} defect")
    return f"{defect_class} defect: {vocabulary}"


class QueryBuilder:
    """Builds one RetrievalQuery per unique defect class from Phase 1 detections."""

    def build(self, detections: list[dict[str, Any]]) -> list[RetrievalQuery]:
        """Convert detection dicts into deduplicated, confidence-ranked RetrievalQuery objects."""
        if not detections:
            return []

        # Merge duplicate classes: keep only the highest-confidence detection per class.
        best_by_class: dict[str, dict[str, Any]] = {}
        for det in detections:
            cls = str(det["defect_class"])
            conf = float(det["confidence"])
            if cls not in best_by_class or conf > best_by_class[cls]["confidence"]:
                best_by_class[cls] = det

        queries: list[RetrievalQuery] = []
        for cls, det in best_by_class.items():
            confidence = float(det["confidence"])
            band = _confidence_band(confidence)
            query = RetrievalQuery(
                defect_class=cls,
                confidence=confidence,
                confidence_band=band,
                query_text=_build_query_text(cls),
                is_uncertain=band == "uncertain",
            )
            logger.debug(
                "Built query | defect_class=%s confidence=%.4f band=%s query_text=%r",
                query.defect_class,
                query.confidence,
                query.confidence_band,
                query.query_text,
            )
            queries.append(query)

        return queries
