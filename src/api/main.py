"""FastAPI application for PCB defect detection (Phase 1) and RAG analysis (Phase 3)."""

from __future__ import annotations

import io
import logging
from typing import Final

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

from src.api.schemas import (
    DetectResponse,
    DetectionItem,
    HealthResponse,
    ReportHealthResponse,
)
from src.generation.generator import ReportGenerator
from src.generation.schemas import InspectionReport
from src.retrieval.confidence import ConfidenceScorer
from src.retrieval.query_builder import QueryBuilder
from src.retrieval.retriever import PCBRetriever
from src.retrieval.schemas import AnalysisResult, RetrievalResult
from src.vision.constants import DEFAULT_BEST_WEIGHTS
from src.vision.detect import DefectDetector, Detection

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES: Final[set[str]] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}
ALLOWED_EXTENSIONS: Final[set[str]] = {".jpg", ".jpeg", ".png"}

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

app = FastAPI(
    title="PCB Defect Detection API",
    description="Phase 1 vision + Phase 3 RAG retrieval for PCB defect inspection",
    version="0.3.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root URL to interactive API docs."""
    return RedirectResponse(url="/docs")

# Module-level singleton; reloaded automatically when weights file changes on disk.
_detector: DefectDetector | None = None
_detector_weights_mtime: float = -1.0


def get_detector(
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
) -> DefectDetector | None:
    """Return cached detector, reloading when weights file changes (e.g. after training).

    iou_threshold is the NMS IoU cutoff: a HIGHER value suppresses fewer overlapping
    boxes, so nearby small defects (e.g. adjacent pads) are less likely to be merged
    away — a recall lever. Lower it back toward the 0.45 default if duplicates appear.
    """
    global _detector, _detector_weights_mtime  # noqa: PLW0603
    if not DEFAULT_BEST_WEIGHTS.is_file():
        return None
    mtime = DEFAULT_BEST_WEIGHTS.stat().st_mtime
    if _detector is not None and mtime == _detector_weights_mtime:
        # Update thresholds on the cached detector if the caller changed them.
        _detector.conf_threshold = conf_threshold
        _detector.iou_threshold = iou_threshold
        return _detector
    logger.info("Loading detector from %s", DEFAULT_BEST_WEIGHTS)
    _detector = DefectDetector(
        DEFAULT_BEST_WEIGHTS,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    _detector_weights_mtime = mtime
    return _detector


# PCBRetriever and ConfidenceScorer are instantiated ONCE here, at module
# load / process startup — not inside the request handler. If ChromaDB is
# unavailable, log a warning and leave _retriever as None; /analyze then
# returns a 503 with a clear message rather than crashing the process.
_query_builder = QueryBuilder()
_scorer = ConfidenceScorer()
try:
    _retriever: PCBRetriever | None = PCBRetriever()
except Exception as exc:
    logger.warning("Could not initialise PCBRetriever at startup: %s", exc)
    _retriever = None

# Phase 4: instantiate the report generator once at startup. It probes local Ollama
# and degrades to fallback templates if unavailable — it never raises at construction.
_report_generator = ReportGenerator()
if _report_generator.llm_available:
    logger.info("Report generator ready | LLM model=%s", _report_generator.model)
else:
    logger.warning("LLM unavailable — /report will use fallback templates")


def _best_detection_by_class(detections: list[Detection]) -> dict[str, Detection]:
    """Return the highest-confidence Detection per class, mirroring QueryBuilder's merge rule."""
    best: dict[str, Detection] = {}
    for det in detections:
        if det.defect_class not in best or det.confidence > best[det.defect_class].confidence:
            best[det.defect_class] = det
    return best


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report model load status without triggering a slow model load."""
    # Check if the model is already loaded in memory, or if the weights exist on disk to be loaded.
    # This prevents the initial health check from timing out on a 3-second timeout limit.
    model_loaded = (_detector is not None) or DEFAULT_BEST_WEIGHTS.is_file()
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/detect", response_model=DetectResponse)
async def detect_defects(
    file: UploadFile = File(...),
    conf: float = Query(0.15, ge=0.01, le=1.0, description="Minimum confidence threshold (lower catches more defects)"),
    iou: float = Query(0.45, ge=0.1, le=0.95, description="NMS IoU threshold (higher keeps more overlapping boxes → recall)"),
) -> DetectResponse:
    """Detect PCB defects in an uploaded image and return bounding boxes."""
    # Validate extension and content-type before any expensive operations.
    _validate_upload(file)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    _validate_magic_bytes(contents)

    # Model check after input validation so bad requests always get 4xx.
    detector = get_detector(conf_threshold=conf, iou_threshold=iou)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train first; expected weights at {DEFAULT_BEST_WEIGHTS}",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file") from exc

    result = detector.detect(image)
    return DetectResponse(
        detections=[
            DetectionItem(
                defect_class=d.defect_class,
                confidence=d.confidence,
                bbox=d.bbox,
            )
            for d in result.detections
        ],
        image_width=result.image_width,
        image_height=result.image_height,
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    conf: float = Query(0.15, ge=0.01, le=1.0, description="Minimum confidence threshold (lower catches more defects)"),
    iou: float = Query(0.45, ge=0.1, le=0.95, description="NMS IoU threshold (higher keeps more overlapping boxes → recall)"),
) -> AnalysisResult:
    """Detect defects, then retrieve matching historical cases and standards with confidence scoring."""
    # Same validation pattern as /detect.
    _validate_upload(file)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    _validate_magic_bytes(contents)

    # Same model-loaded check as /detect, same error response shape.
    detector = get_detector(conf_threshold=conf, iou_threshold=iou)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train first; expected weights at {DEFAULT_BEST_WEIGHTS}",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file") from exc

    detection_result = detector.detect(image)

    # Zero detections is a clean, valid result — return immediately, no ChromaDB touch.
    if not detection_result.detections:
        return AnalysisResult(total_detections=0, results=[])

    # Only check retrieval availability once we know retrieval will actually be used.
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieval database not available. "
                "Run ingest_cases.py and ingest_standards.py first."
            ),
        )

    detection_dicts = [
        {"defect_class": d.defect_class, "confidence": d.confidence, "bbox": d.bbox}
        for d in detection_result.detections
    ]
    queries = _query_builder.build(detection_dicts)
    best_detection_by_class = _best_detection_by_class(detection_result.detections)

    results: list[RetrievalResult] = []
    for query in queries:
        detection = best_detection_by_class[query.defect_class]
        cases = _retriever.retrieve_cases(query)
        standards = _retriever.retrieve_standards(query.defect_class)
        result = _retriever.combine(detection, cases, standards)
        result = _scorer.score(result)
        results.append(result)

    return AnalysisResult(total_detections=len(results), results=results)


def _run_retrieval_pipeline(detection_result) -> AnalysisResult:
    """Run the Phase 3 retrieval + scoring pipeline over a detection result (mirrors /analyze)."""
    # Zero detections is a clean, valid result — no ChromaDB touch.
    if not detection_result.detections:
        return AnalysisResult(total_detections=0, results=[])

    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieval database not available. "
                "Run ingest_cases.py and ingest_standards.py first."
            ),
        )

    detection_dicts = [
        {"defect_class": d.defect_class, "confidence": d.confidence, "bbox": d.bbox}
        for d in detection_result.detections
    ]
    queries = _query_builder.build(detection_dicts)
    best_detection_by_class = _best_detection_by_class(detection_result.detections)

    results: list[RetrievalResult] = []
    for query in queries:
        detection = best_detection_by_class[query.defect_class]
        cases = _retriever.retrieve_cases(query)
        standards = _retriever.retrieve_standards(query.defect_class)
        result = _retriever.combine(detection, cases, standards)
        result = _scorer.score(result)
        results.append(result)

    return AnalysisResult(total_detections=len(results), results=results)


@app.get("/report/health", response_model=ReportHealthResponse)
def report_health() -> ReportHealthResponse:
    """Report whether the LLM is available, so the frontend can show a fallback-mode indicator."""
    available = _report_generator.llm_available
    return ReportHealthResponse(
        llm_available=available,
        model=_report_generator.model if available else None,
        fallback_mode=not available,
    )


@app.post("/report", response_model=InspectionReport)
async def generate_report_endpoint(
    file: UploadFile = File(...),
    conf: float = Query(0.15, ge=0.01, le=1.0, description="Minimum confidence threshold (lower catches more defects)"),
    iou: float = Query(0.45, ge=0.1, le=0.95, description="NMS IoU threshold (higher keeps more overlapping boxes → recall)"),
) -> InspectionReport:
    """Detect, retrieve, score, then generate a structured LLM inspection report (fallback-safe)."""
    # Same validation pattern as /detect and /analyze.
    _validate_upload(file)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    _validate_magic_bytes(contents)

    detector = get_detector(conf_threshold=conf, iou_threshold=iou)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train first; expected weights at {DEFAULT_BEST_WEIGHTS}",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file") from exc

    detection_result = detector.detect(image)
    analysis = _run_retrieval_pipeline(detection_result)
    # Pass the true image dimensions so bbox->location is accurate for any input size.
    return _report_generator.generate_report(
        analysis,
        image_width=detection_result.image_width,
        image_height=detection_result.image_height,
    )


def _validate_upload(file: UploadFile) -> None:
    """Reject uploads with a disallowed extension or content type.

    Note: the magic-bytes check (_validate_magic_bytes) that runs after reading
    the file body is the authoritative guard.  This pre-check catches obviously
    wrong uploads early, but is intentionally lenient — browsers may send
    'application/octet-stream' or append charset parameters to valid images.
    """
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

    # Extract base MIME type, stripping parameters like '; charset=utf-8'.
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    # Allow empty, generic, or known image types — magic bytes are the real guard.
    passthrough_types = {"", "application/octet-stream"}
    if content_type and content_type not in ALLOWED_CONTENT_TYPES and content_type not in passthrough_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{content_type}'. Allowed: JPEG, PNG",
        )


def _validate_magic_bytes(contents: bytes) -> None:
    """Validate that the file is a readable image (JPEG or PNG).

    Instead of only checking raw magic bytes, attempt to open with PIL.
    This handles edge cases like JFIF/EXIF/progressive JPEGs, images saved by
    screenshot tools with non-standard headers, and similar quirks.
    """
    # Fast path: obvious JPEG or PNG magic bytes.
    if contents[:3] == _JPEG_MAGIC or contents[:8] == _PNG_MAGIC:
        return
    # Slow path: let PIL try to identify the format.
    try:
        img = Image.open(io.BytesIO(contents))
        fmt = (img.format or "").upper()
        if fmt in {"JPEG", "PNG", "MPO"}:
            return
    except Exception:
        pass
    raise HTTPException(
        status_code=415,
        detail="File content does not match a supported image format (JPEG or PNG).",
    )
