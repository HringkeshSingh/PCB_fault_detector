"""
Phase 3 retrieval pipeline tests.

Run with:  pytest tests/test_retrieval.py -v

Tests cover QueryBuilder, PCBRetriever, ConfidenceScorer, and the /analyze endpoint.
  - Unit tests (query builder, confidence scorer) — no external dependencies, always runnable.
  - Integration tests (ChromaDB queries) — marked with @pytest.mark.integration,
    skipped unless the ChromaDB directory exists on disk.

Run only unit tests:      pytest tests/test_retrieval.py -v -m "not integration"
Run all including chroma: pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.retrieval.confidence import ConfidenceScorer
from src.retrieval.query_builder import QueryBuilder
from src.retrieval.schemas import (
    AnalysisResult,
    CaseResult,
    RetrievalMetadata,
    RetrievalResult,
    StandardResult,
)
from src.vision.detect import Detection

# ===========================================================================
# Fixtures & Constants
# ===========================================================================

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
CHROMA_AVAILABLE = (CHROMA_DIR / "chroma.sqlite3").is_file()

requires_chroma = pytest.mark.skipif(
    not CHROMA_AVAILABLE,
    reason="ChromaDB not found — run ingest_cases.py and ingest_standards.py first",
)
integration = pytest.mark.integration


def _make_detection_dict(defect_class: str, confidence: float, bbox=None) -> dict:
    """Helper to construct a detection dict for QueryBuilder.build()."""
    d = {"defect_class": defect_class, "confidence": confidence}
    if bbox is not None:
        d["bbox"] = bbox
    return d


def _make_case_result(
    case_id: str = "PCB-CASE-0001",
    defect_class: str = "open",
    similarity: float = 0.85,
) -> CaseResult:
    """Helper to construct a CaseResult for testing."""
    return CaseResult(
        case_id=case_id,
        defect_class=defect_class,
        root_cause="Under-etch left thin copper neck",
        corrective_action="Recalibrated etch spray pressure",
        severity=3,
        outcome_notes="Etch rate verified hourly; no recurrence",
        similarity_score=similarity,
    )


def _make_detection(defect_class: str, confidence: float, bbox=None) -> Detection:
    """Helper to construct a Detection for testing."""
    if bbox is None:
        bbox = [10.0, 10.0, 20.0, 20.0]
    return Detection(defect_class=defect_class, confidence=confidence, bbox=bbox)


# ===========================================================================
# UNIT TESTS — QueryBuilder
# ===========================================================================


class TestQueryBuilder:
    """Test QueryBuilder.build() with various inputs."""

    def test_zero_detections_returns_empty_list(self):
        qb = QueryBuilder()
        result = qb.build([])
        assert result == []

    def test_single_detection_produces_one_query(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.82)])
        assert len(result) == 1
        q = result[0]
        assert q.defect_class == "open"
        assert q.confidence == 0.82
        assert q.confidence_band == "high"
        assert q.is_uncertain is False

    def test_multi_defect_produces_separate_queries(self):
        qb = QueryBuilder()
        result = qb.build([
            _make_detection_dict("open", 0.71),
            _make_detection_dict("short", 0.38),
        ])
        assert len(result) == 2
        classes = {q.defect_class for q in result}
        assert classes == {"open", "short"}

    def test_duplicate_class_keeps_highest_confidence(self):
        qb = QueryBuilder()
        result = qb.build([
            _make_detection_dict("open", 0.55),
            _make_detection_dict("open", 0.91),
        ])
        assert len(result) == 1
        assert result[0].confidence == 0.91

    def test_confidence_band_high(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.82)])
        assert result[0].confidence_band == "high"
        assert result[0].is_uncertain is False

    def test_confidence_band_medium(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.55)])
        assert result[0].confidence_band == "medium"
        assert result[0].is_uncertain is False

    def test_confidence_band_low(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("short", 0.38)])
        assert result[0].confidence_band == "low"
        assert result[0].is_uncertain is False

    def test_confidence_band_uncertain(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("short", 0.15)])
        assert result[0].confidence_band == "uncertain"
        assert result[0].is_uncertain is True

    def test_boundary_0_70_is_high_not_medium(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.70)])
        assert result[0].confidence_band == "high"

    def test_boundary_0_40_is_medium_not_low(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.40)])
        assert result[0].confidence_band == "medium"

    def test_boundary_0_20_is_low_not_uncertain(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.20)])
        assert result[0].confidence_band == "low"

    def test_query_text_contains_defect_class_and_vocabulary(self):
        qb = QueryBuilder()
        result = qb.build([_make_detection_dict("open", 0.82)])
        query_text = result[0].query_text
        assert "open" in query_text.lower()
        assert "defect" in query_text.lower()
        # Verify vocabulary terms from the class vocabulary dict are present
        assert any(term in query_text.lower() for term in ["etch", "plating", "trace"])

    def test_each_defect_class_has_unique_vocabulary(self):
        qb = QueryBuilder()
        result = qb.build([
            _make_detection_dict("open", 0.82),
            _make_detection_dict("short", 0.75),
        ])
        open_text = result[0].query_text.lower()
        short_text = result[1].query_text.lower()
        # Should have different vocabulary
        assert "solder" in short_text  # short-specific
        assert "etch" in open_text or "plating" in open_text  # open-specific


# ===========================================================================
# UNIT TESTS — ConfidenceScorer
# ===========================================================================


class TestConfidenceScorer:
    """Test ConfidenceScorer.score() logic."""

    def test_high_confidence_with_high_similarity(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.75)
        cases = [_make_case_result(similarity=0.85)]
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=1,
            standards_found=0,
            top_case_similarity=0.85,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=cases,
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.retrieval_confidence == "high"

    def test_uncertain_when_detection_confidence_too_low(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.35)
        cases = [_make_case_result(similarity=0.95)]
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=1,
            standards_found=0,
            top_case_similarity=0.95,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=cases,
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.retrieval_confidence == "uncertain"
        assert result.retrieval_metadata.flagged_for_human_review is True

    def test_low_confidence_when_no_cases_found(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.75)
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=0,
            standards_found=0,
            top_case_similarity=0.0,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=[],
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.top_case_similarity == 0.0
        assert result.retrieval_metadata.flagged_for_human_review is True

    def test_flagged_when_too_few_cases(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.75)
        cases = [_make_case_result(similarity=0.85)]
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=1,
            standards_found=0,
            top_case_similarity=0.85,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=cases,
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.flagged_for_human_review is True

    def test_flagged_when_similarity_too_low(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.75)
        cases = [
            _make_case_result(similarity=0.50),
            _make_case_result(similarity=0.45),
            _make_case_result(similarity=0.40),
        ]
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=3,
            standards_found=0,
            top_case_similarity=0.50,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=cases,
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.flagged_for_human_review is True

    def test_not_flagged_when_all_good(self):
        scorer = ConfidenceScorer()
        det = _make_detection("open", 0.75)
        cases = [
            _make_case_result(similarity=0.85),
            _make_case_result(similarity=0.80),
            _make_case_result(similarity=0.75),
        ]
        meta = RetrievalMetadata(
            retrieval_confidence="uncertain",
            flagged_for_human_review=True,
            cases_found=3,
            standards_found=0,
            top_case_similarity=0.85,
            standards_skipped=True,
        )
        result = RetrievalResult(
            detection=det,
            retrieved_cases=cases,
            retrieved_standards=[],
            retrieval_metadata=meta,
        )

        result = scorer.score(result)
        assert result.retrieval_metadata.retrieval_confidence == "high"
        assert result.retrieval_metadata.flagged_for_human_review is False


# ===========================================================================
# UNIT TESTS — PCBRetriever (mocked ChromaDB)
# ===========================================================================


class TestPCBRetrieverUnit:
    """Test PCBRetriever with mocked ChromaDB."""

    def _make_raw_chroma_response(self, n: int, defect_class: str) -> dict:
        """Synthetic ChromaDB query response."""
        metadatas = [[
            {
                "case_id": f"PCB-CASE-{i:04d}",
                "defect_class": defect_class,
                "root_cause": f"Root cause {i}",
                "corrective_action": f"Action {i}",
                "severity": 3,
                "outcome_notes": f"Outcome {i}",
            }
            for i in range(n)
        ]]
        distances = [[0.10 + i * 0.05 for i in range(n)]]
        return {"metadatas": metadatas, "distances": distances}

    def test_parse_case_results_correct_count(self):
        from src.retrieval.retriever import _parse_case_results

        raw = self._make_raw_chroma_response(3, "open")
        cases = _parse_case_results(raw)
        assert len(cases) == 3
        assert all(c.defect_class == "open" for c in cases)

    def test_similarity_score_derived_from_distance(self):
        from src.retrieval.retriever import _parse_case_results

        raw = self._make_raw_chroma_response(1, "open")
        raw["distances"] = [[0.2]]
        cases = _parse_case_results(raw)
        assert abs(cases[0].similarity_score - 0.8) < 0.001

    def test_retriever_initialization_requires_chroma_dir(self):
        from src.retrieval.retriever import PCBRetriever

        with pytest.raises(FileNotFoundError):
            PCBRetriever(chroma_dir=Path("/nonexistent/path"))


# ===========================================================================
# INTEGRATION TESTS — real ChromaDB
# ===========================================================================


@integration
@requires_chroma
class TestPCBRetrieverIntegration:
    """Tests against the real ChromaDB. Require ingest_cases.py to have been run."""

    @pytest.fixture(scope="class")
    def retriever(self):
        from src.retrieval.retriever import PCBRetriever

        return PCBRetriever(chroma_dir=CHROMA_DIR)

    def test_retrieve_cases_returns_correct_class_only(self, retriever):
        """Cases returned for a defect class must all belong to that class."""
        from src.retrieval.query_builder import QueryBuilder

        qb = QueryBuilder()
        queries = qb.build([_make_detection_dict("open", 0.82)])
        query = queries[0]

        cases = retriever.retrieve_cases(query)
        assert len(cases) > 0
        for case in cases:
            assert case.defect_class == "open"

    def test_retrieve_standards_returns_results(self, retriever):
        """retrieve_standards() should return StandardResult objects with excerpt."""
        standards = retriever.retrieve_standards("open")
        if not standards:
            pytest.skip("Standards collection empty")
        for std in standards:
            assert std.section_id
            assert std.source_doc
            assert std.excerpt
            assert 0.0 <= std.relevance_score <= 1.0

    def test_combine_produces_retrieval_result(self, retriever):
        """combine() assembles a complete RetrievalResult with placeholder fields."""
        from src.retrieval.query_builder import QueryBuilder

        qb = QueryBuilder()
        queries = qb.build([_make_detection_dict("open", 0.82)])
        query = queries[0]

        det = _make_detection("open", 0.82)
        cases = retriever.retrieve_cases(query)
        standards = retriever.retrieve_standards("open")
        result = retriever.combine(det, cases, standards)

        assert isinstance(result, RetrievalResult)
        assert result.detection.defect_class == "open"
        assert len(result.retrieved_cases) == len(cases)
        assert len(result.retrieved_standards) == len(standards)
        assert result.retrieval_metadata.retrieval_confidence == "uncertain"
        assert result.retrieval_metadata.flagged_for_human_review is True

    @pytest.mark.parametrize("defect_class", [
        "open", "short", "mousebite", "spur", "copper", "pin-hole",
    ])
    def test_each_class_retrieves_at_least_one_case(self, retriever, defect_class):
        """Each supported defect class should return at least one case."""
        from src.retrieval.query_builder import QueryBuilder

        qb = QueryBuilder()
        queries = qb.build([_make_detection_dict(defect_class, 0.75)])
        query = queries[0]

        cases = retriever.retrieve_cases(query)
        assert len(cases) >= 1, f"No cases returned for class '{defect_class}'"

    def test_similarity_scores_bounded_zero_one(self, retriever):
        """All similarity scores must be between 0 and 1."""
        from src.retrieval.query_builder import QueryBuilder

        qb = QueryBuilder()
        queries = qb.build([_make_detection_dict("pin-hole", 0.70)])
        query = queries[0]

        cases = retriever.retrieve_cases(query)
        for case in cases:
            assert 0.0 <= case.similarity_score <= 1.0


# ===========================================================================
# INTEGRATION TESTS — /analyze endpoint
# ===========================================================================


@integration
@requires_chroma
class TestAnalyzeEndpoint:
    """Tests the FastAPI /analyze endpoint."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        return TestClient(app)

    def test_zero_detections_returns_empty_result(self, client):
        """Send a real image to /analyze but fake no detections."""
        from unittest.mock import patch

        from src.vision.detect import DetectionResult

        with patch("src.api.main.get_detector") as mock_detector_fn:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = DetectionResult(
                detections=[],
                image_width=640,
                image_height=640,
            )
            mock_detector_fn.return_value = mock_detector

            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (10, 10), (128, 128, 128)).save(buf, format="JPEG")
            buf.seek(0)

            response = client.post(
                "/analyze", files={"file": ("test.jpg", buf.read(), "image/jpeg")}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["total_detections"] == 0
            assert body["results"] == []

    def test_single_detection_produces_result(self, client):
        """Send a real image with one detection to /analyze."""
        from unittest.mock import patch

        from src.vision.detect import Detection, DetectionResult

        with patch("src.api.main.get_detector") as mock_detector_fn:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = DetectionResult(
                detections=[
                    Detection(defect_class="open", confidence=0.82, bbox=[10, 10, 20, 20])
                ],
                image_width=640,
                image_height=640,
            )
            mock_detector_fn.return_value = mock_detector

            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (10, 10), (128, 128, 128)).save(buf, format="JPEG")
            buf.seek(0)

            response = client.post(
                "/analyze", files={"file": ("test.jpg", buf.read(), "image/jpeg")}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["total_detections"] == 1
            assert len(body["results"]) == 1
            result = body["results"][0]
            assert result["detection"]["defect_class"] == "open"
            assert result["detection"]["confidence"] == 0.82
            assert "retrieval_metadata" in result
            assert "retrieved_cases" in result
            assert "retrieved_standards" in result

    def test_analysis_result_matches_schema(self, client):
        """Verify /analyze response matches AnalysisResult schema."""
        from unittest.mock import patch

        from src.vision.detect import Detection, DetectionResult

        with patch("src.api.main.get_detector") as mock_detector_fn:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = DetectionResult(
                detections=[
                    Detection(
                        defect_class="open", confidence=0.82, bbox=[10, 10, 20, 20]
                    ),
                    Detection(
                        defect_class="short", confidence=0.65, bbox=[30, 30, 40, 40]
                    ),
                ],
                image_width=640,
                image_height=640,
            )
            mock_detector_fn.return_value = mock_detector

            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (10, 10), (128, 128, 128)).save(buf, format="JPEG")
            buf.seek(0)

            response = client.post(
                "/analyze", files={"file": ("test.jpg", buf.read(), "image/jpeg")}
            )
            assert response.status_code == 200
            body = response.json()

            # Verify top-level structure
            assert "total_detections" in body
            assert "results" in body
            assert isinstance(body["results"], list)

            # Verify each result has required fields
            for result in body["results"]:
                assert "detection" in result
                assert "retrieved_cases" in result
                assert "retrieved_standards" in result
                assert "retrieval_metadata" in result

                # Verify detection structure
                det = result["detection"]
                assert "defect_class" in det
                assert "confidence" in det
                assert "bbox" in det

                # Verify retrieval_metadata structure
                meta = result["retrieval_metadata"]
                assert "retrieval_confidence" in meta
                assert "flagged_for_human_review" in meta
                assert "cases_found" in meta
                assert "standards_found" in meta
                assert "top_case_similarity" in meta
                assert "standards_skipped" in meta

    def test_file_validation_rejects_non_image(self, client):
        """POST /analyze with a non-image file should be rejected."""
        response = client.post(
            "/analyze", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 415

    def test_file_validation_rejects_bad_magic_bytes(self, client):
        """POST /analyze with fake image bytes should be rejected."""
        response = client.post(
            "/analyze",
            files={"file": ("fake.jpg", b"not an image at all", "image/jpeg")},
        )
        assert response.status_code == 415
