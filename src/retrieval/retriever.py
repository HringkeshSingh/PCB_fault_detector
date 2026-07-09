"""
Phase 3 core retriever.

Queries ChromaDB for historical cases and standards excerpts and assembles
them into a RetrievalResult. Confidence scoring is deliberately NOT done
here — that is confidence.py's responsibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.data.ingest_cases import (
    COLLECTION_NAME as CASES_COLLECTION,
    DEFAULT_CHROMA_DIR,
    EMBEDDING_MODEL,
)
from src.retrieval.schemas import CaseResult, RetrievalMetadata, RetrievalQuery, RetrievalResult, StandardResult
from src.vision.detect import Detection

# Source of truth for the standards collection name — ingest_standards.py imports this.
STANDARDS_COLLECTION: str = "pcb_standards"

logger = logging.getLogger(__name__)


def _distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB cosine distance into a bounded [0, 1] similarity score."""
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


def _parse_case_results(raw: dict[str, Any]) -> list[CaseResult]:
    """Turn a raw ChromaDB cases-collection query response into CaseResult objects."""
    cases: list[CaseResult] = []
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for meta, dist in zip(metadatas, distances):
        if meta is None:
            continue
        cases.append(
            CaseResult(
                case_id=meta.get("case_id", ""),
                defect_class=meta.get("defect_class", ""),
                component_type=meta.get("component_type", ""),
                date_recorded=meta.get("date_recorded", ""),
                root_cause=meta.get("root_cause", ""),
                corrective_action=meta.get("corrective_action", ""),
                severity=int(meta.get("severity", 3)),
                outcome_notes=meta.get("outcome_notes", ""),
                similarity_score=_distance_to_similarity(dist),
            )
        )
    return cases


def _parse_standard_results(raw: dict[str, Any]) -> list[StandardResult]:
    """Turn a raw ChromaDB standards-collection query response into StandardResult objects."""
    standards: list[StandardResult] = []
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for meta, dist in zip(metadatas, distances):
        if meta is None:
            continue
        standards.append(
            StandardResult(
                section_id=meta.get("section_id", ""),
                source_doc=meta.get("source_doc", ""),
                excerpt=meta.get("excerpt", ""),
                relevance_score=_distance_to_similarity(dist),
            )
        )
    return standards


class PCBRetriever:
    """Queries ChromaDB for historical cases and standards excerpts."""

    def __init__(
        self,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        top_k_cases: int = 5,
        top_k_standards: int = 3,
    ) -> None:
        """Load the ChromaDB client and embedding model once; probe collection availability."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_dir = chroma_dir.resolve()
        if not chroma_dir.is_dir():
            raise FileNotFoundError(
                f"ChromaDB directory not found: {chroma_dir}. Run ingest_cases.py first."
            )

        self.top_k_cases = top_k_cases
        self.top_k_standards = top_k_standards

        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._model = SentenceTransformer(embedding_model)

        # Cases collection is required — fail loudly if missing.
        self._cases_collection = self._client.get_collection(CASES_COLLECTION)

        # Standards collection is optional. standards_available is the single
        # source of truth combine() uses to set standards_skipped.
        try:
            self._standards_collection = self._client.get_collection(STANDARDS_COLLECTION)
            if self._standards_collection.count() == 0:
                logger.warning("Standards collection '%s' exists but is empty", STANDARDS_COLLECTION)
                self.standards_available = False
            else:
                self.standards_available = True
        except Exception:
            logger.warning("Standards collection '%s' not found — standards retrieval disabled", STANDARDS_COLLECTION)
            self._standards_collection = None
            self.standards_available = False

    def retrieve_cases(self, query: RetrievalQuery) -> list[CaseResult]:
        """Query pcb_defect_cases for the given RetrievalQuery, pre-filtered by defect_class."""
        embedding = self._model.encode([query.query_text]).tolist()
        where_filter = {"defect_class": {"$eq": query.defect_class}}

        raw = self._cases_collection.query(
            query_embeddings=embedding,
            n_results=self.top_k_cases,
            where=where_filter,
            include=["metadatas", "distances"],
        )
        cases = _parse_case_results(raw)

        top_sim = cases[0].similarity_score if cases else None
        logger.info(
            "retrieve_cases | collection=%s | filter=%s | top_k=%d | results=%d | top_similarity=%s",
            CASES_COLLECTION,
            where_filter,
            self.top_k_cases,
            len(cases),
            f"{top_sim:.4f}" if top_sim is not None else "n/a",
        )
        return cases

    def retrieve_standards(self, defect_class: str) -> list[StandardResult]:
        """Query pcb_standards for the given defect_class; returns [] if unavailable."""
        if not self.standards_available:
            logger.info(
                "retrieve_standards | collection=%s | filter=defect_class=%s | skipped=True (collection unavailable/empty)",
                STANDARDS_COLLECTION,
                defect_class,
            )
            return []

        query_text = (
            f"{defect_class} defect inspection standard: acceptance criteria, "
            f"corrective action, and re-inspection requirement"
        )
        embedding = self._model.encode([query_text]).tolist()
        where_filter = {"defect_class": {"$eq": defect_class}}

        raw = self._standards_collection.query(
            query_embeddings=embedding,
            n_results=self.top_k_standards,
            where=where_filter,
            include=["metadatas", "distances"],
        )
        standards = _parse_standard_results(raw)

        top_rel = standards[0].relevance_score if standards else None
        logger.info(
            "retrieve_standards | collection=%s | filter=%s | top_k=%d | results=%d | top_similarity=%s",
            STANDARDS_COLLECTION,
            where_filter,
            self.top_k_standards,
            len(standards),
            f"{top_rel:.4f}" if top_rel is not None else "n/a",
        )
        return standards

    def combine(
        self,
        detection: Detection,
        cases: list[CaseResult],
        standards: list[StandardResult],
    ) -> RetrievalResult:
        """Assemble a RetrievalResult; leaves confidence fields as placeholders for confidence.py."""
        top_case_similarity = cases[0].similarity_score if cases else 0.0

        metadata = RetrievalMetadata(
            retrieval_confidence="uncertain",  # placeholder — ConfidenceScorer.score() overwrites
            flagged_for_human_review=True,  # placeholder — ConfidenceScorer.score() overwrites
            cases_found=len(cases),
            standards_found=len(standards),
            top_case_similarity=top_case_similarity,
            standards_skipped=not self.standards_available,
        )
        return RetrievalResult(
            detection=detection,
            retrieved_cases=cases,
            retrieved_standards=standards,
            retrieval_metadata=metadata,
        )
