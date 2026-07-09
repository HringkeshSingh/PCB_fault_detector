"""Shared constants for DeepPCB defect detection."""

from pathlib import Path

# Project root: PCB_faultDetection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DeepPCB integer type ID -> defect name (from dataset README)
DEEPPCB_CLASS_ID_TO_NAME: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

# YOLO class index (0-based) uses the same order as DEEPPCB IDs 1-6
CLASS_NAMES: list[str] = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]

# DeepPCB IDs are 1-6; YOLO indices are 0-5 (deeppcb_id - 1).
DEEPPCB_ID_TO_YOLO_IDX: dict[int, int] = {
    deeppcb_id: deeppcb_id - 1 for deeppcb_id in DEEPPCB_CLASS_ID_TO_NAME
}

YOLO_IDX_TO_CLASS_NAME: dict[int, str] = {
    idx: name for idx, name in enumerate(CLASS_NAMES)
}

# Default paths (override via CLI or env)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "DeepPCB" / "PCBData"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_BEST_WEIGHTS = DEFAULT_MODELS_DIR / "best.pt"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

# Per-class confidence thresholds for detection post-filtering.
# Classes with high precision but low recall (mousebite, spur, copper) use a
# lower threshold to recover missed detections.  The downstream confidence
# scorer (src/retrieval/confidence.py) still flags low-confidence detections
# for human review, so lowering these thresholds does not silently degrade
# report quality.
PER_CLASS_CONF_THRESHOLD: dict[str, float] = {
    "open": 0.20,
    "short": 0.20,
    "mousebite": 0.10,
    "spur": 0.10,
    "copper": 0.10,
    "pin-hole": 0.25,
}

# Fallback threshold for any class not in PER_CLASS_CONF_THRESHOLD.
DEFAULT_CONF_THRESHOLD: float = 0.15

# VOC_PCB (real-photo fine-tuning dataset) — see prepare_voc_pcb.py.
# Class names differ from DeepPCB's; map to the same 6-class taxonomy above.
VOC_PCB_CLASS_NAME_MAP: dict[str, str] = {
    "open_circuit": "open",
    "short": "short",
    "mouse_bite": "mousebite",
    "spur": "spur",
    "spurious_copper": "copper",
    "missing_hole": "pin-hole",
}

DEFAULT_VOC_ZIP = PROJECT_ROOT / "VOC_PCB.zip"
DEFAULT_VOC_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "VOC_PCB"
DEFAULT_FINETUNE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_finetune"
