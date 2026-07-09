"""Inference module for PCB defect detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from src.vision.constants import (
    DEFAULT_BEST_WEIGHTS,
    DEFAULT_CONF_THRESHOLD,
    PER_CLASS_CONF_THRESHOLD,
    YOLO_IDX_TO_CLASS_NAME,
)

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, np.ndarray]


class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single defect detection."""

    defect_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")


class DetectionResult(BaseModel):
    """Structured inference output."""

    detections: list[Detection]
    image_width: int
    image_height: int


class DefectDetector:
    """Loads a YOLOv8 model and runs defect detection."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_BEST_WEIGHTS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = 0.45,
    ) -> None:
        from ultralytics import YOLO

        weights_path = Path(weights_path).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. Train with train.py first."
            )
        self._model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: ImageInput) -> DetectionResult:
        """Run inference on a file path or in-memory image.

        YOLO runs at a low global threshold (self.conf_threshold) so that
        borderline detections for weak classes are not discarded by the
        engine.  _parse_results() then applies per-class thresholds from
        PER_CLASS_CONF_THRESHOLD to filter each detection individually.
        """
        pil_image, width, height = _load_image(image)
        # Use the lower of self.conf_threshold and the per-class floor
        # so YOLO doesn't discard candidates before we can apply
        # per-class filtering.
        inference_conf = min(
            self.conf_threshold,
            min(PER_CLASS_CONF_THRESHOLD.values(), default=self.conf_threshold),
        )
        results = self._model.predict(
            source=pil_image,
            conf=inference_conf,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections = _parse_results(results, self.conf_threshold)
        return DetectionResult(
            detections=detections,
            image_width=width,
            image_height=height,
        )

    def detect_to_dict(self, image: ImageInput) -> dict[str, Any]:
        """Return detection result as a JSON-serializable dict."""
        return self.detect(image).model_dump()


def _load_image(image: ImageInput) -> tuple[Image.Image, int, int]:
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return pil, pil.width, pil.height


def _parse_results(
    results: Any,
    fallback_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> list[Detection]:
    """Parse YOLO results and apply per-class confidence filtering.

    Each detection is kept only if its confidence meets or exceeds the
    threshold for its class (from PER_CLASS_CONF_THRESHOLD), falling back
    to *fallback_threshold* for classes not in that dict.
    """
    detections: list[Detection] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        cls_idx = int(box.cls.item())
        class_name = YOLO_IDX_TO_CLASS_NAME.get(cls_idx)
        if class_name is None:
            logger.warning(
                "Unknown YOLO class index %d returned by model — skipping detection",
                cls_idx,
            )
            continue
        conf = float(box.conf.item())

        # Per-class threshold: use the class-specific value when available,
        # otherwise fall back to the caller-supplied threshold.
        threshold = PER_CLASS_CONF_THRESHOLD.get(class_name, fallback_threshold)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                defect_class=class_name,
                confidence=round(conf, 4),
                bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            )
        )
    return detections


def detect(
    image: ImageInput,
    weights_path: Path = DEFAULT_BEST_WEIGHTS,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> dict[str, Any]:
    """Convenience function: one-shot detection returning a dict.

    Note: constructs a new DefectDetector (full model load) on every call.
    For repeated inference, instantiate and reuse a DefectDetector directly.
    """
    detector = DefectDetector(weights_path, conf_threshold=conf_threshold)
    return detector.detect_to_dict(image)
