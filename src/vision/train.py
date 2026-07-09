"""Fine-tune YOLOv8 on prepared DeepPCB dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Any

from src.vision.constants import (
    CLASS_NAMES,
    DEFAULT_BEST_WEIGHTS,
    DEFAULT_MODELS_DIR,
    DEFAULT_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def train_model(
    data_yaml: Path = DEFAULT_PROCESSED_DIR / "data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: Path = DEFAULT_MODELS_DIR,
    run_name: str = "deeppcb",
    seed: int = 42,
    freeze: int | None = None,
    lr0: float | None = None,
    workers: int | None = None,
    translate: float | None = None,
) -> Path:
    """
    Fine-tune a pretrained YOLOv8 checkpoint on DeepPCB (or another prepared dataset).

    freeze/lr0 are for continuing training from an existing checkpoint (e.g. a
    real-photo fine-tune on top of models/best.pt) — left as None for a normal
    full training run, which uses Ultralytics' defaults unchanged.

    Saves best weights to models/best.pt and metrics to models/training_results.json.
    Raises RuntimeError if neither best.pt nor last.pt exists after training.
    """
    from ultralytics import YOLO

    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. Run prepare_data.py first."
        )

    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        "seed": seed,
        "save": True,
        "plots": True,
    }
    if freeze is not None:
        train_kwargs["freeze"] = freeze
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    if workers is not None:
        train_kwargs["workers"] = workers
    if translate is not None:
        train_kwargs["translate"] = translate

    model = YOLO(model_name)
    results = model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    _verify_epochs_completed(run_dir, epochs)

    best_src = run_dir / "weights" / "best.pt"
    checkpoint_used = "best.pt"
    if not best_src.is_file():
        last_src = run_dir / "weights" / "last.pt"
        if not last_src.is_file():
            raise RuntimeError(
                f"Neither best.pt nor last.pt found under {run_dir / 'weights'}. "
                "Training may have crashed before saving any checkpoint."
            )
        warnings.warn(
            f"best.pt not found in {run_dir / 'weights'}; falling back to last.pt. "
            "The deployed model may be a suboptimal last-epoch checkpoint.",
            stacklevel=2,
        )
        best_src = last_src
        checkpoint_used = "last.pt"

    best_dst = DEFAULT_BEST_WEIGHTS
    best_dst.parent.mkdir(parents=True, exist_ok=True)
    best_dst.write_bytes(best_src.read_bytes())

    metrics = _extract_metrics(model, data_yaml, run_dir)
    metrics["checkpoint_used"] = checkpoint_used
    results_path = project / "training_results.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Best weights: %s", best_dst)
    logger.info("Metrics: %s", results_path)
    if checkpoint_used != "best.pt":
        logger.warning("Checkpoint used: %s (best.pt was absent)", checkpoint_used)
    return best_dst


def _verify_epochs_completed(run_dir: Path, expected_epochs: int) -> None:
    """Warn if results.csv has fewer rows than expected, indicating early stop or crash."""
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        logger.warning("results.csv not found in %s — cannot verify epoch completion", run_dir)
        return
    with results_csv.open(newline="") as f:
        actual_epochs = sum(1 for _ in csv.reader(f)) - 1  # subtract header row
    # Allow up to 10 epochs short (Ultralytics early-stopping patience default is 50).
    if actual_epochs < expected_epochs - 10:
        warnings.warn(
            f"Training ran only {actual_epochs} epochs (expected {expected_epochs}). "
            "Check for early stopping, OOM, or interrupted training.",
            stacklevel=3,
        )
    logger.info("Epochs completed: %d / %d", actual_epochs, expected_epochs)


def _extract_metrics(
    model: Any,
    data_yaml: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Run validation and collect overall + per-class mAP, precision, and recall."""
    val_results = model.val(data=str(data_yaml), split="val")
    box = getattr(val_results, "box", None)

    # Detect validation failure: key scalar attributes missing or None.
    map50_val = getattr(box, "map50", None) if box is not None else None
    if map50_val is None:
        logger.error("Validation metrics unavailable — box.map50 attribute missing")
        return {
            "status": "validation_failed",
            "class_names": CLASS_NAMES,
            "run_dir": str(run_dir),
        }

    metrics: dict[str, Any] = {
        "status": "ok",
        "class_names": CLASS_NAMES,
        "run_dir": str(run_dir),
        "map50": float(map50_val),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(getattr(box, "mp", 0.0) or 0.0),
        "recall": float(getattr(box, "mr", 0.0) or 0.0),
        "per_class": {},
    }

    # Per-class precision and recall (shape: (nc,) numpy arrays).
    per_class_p = getattr(box, "p", None)
    per_class_r = getattr(box, "r", None)
    per_class_ap50 = getattr(box, "ap50", None)
    per_class_maps = getattr(box, "maps", None)

    for idx, name in enumerate(CLASS_NAMES):
        entry: dict[str, float] = {}
        if per_class_p is not None and idx < len(per_class_p):
            entry["precision"] = float(per_class_p[idx])
        if per_class_r is not None and idx < len(per_class_r):
            entry["recall"] = float(per_class_r[idx])
            if entry.get("recall", 1.0) < 0.5:
                logger.warning(
                    "Low recall for class '%s': %.3f — check labels and class balance",
                    name,
                    entry["recall"],
                )
        if per_class_ap50 is not None and idx < len(per_class_ap50):
            entry["ap50"] = float(per_class_ap50[idx])
        if per_class_maps is not None and idx < len(per_class_maps):
            entry["map50_95"] = float(per_class_maps[idx])
        metrics["per_class"][name] = entry

    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        metrics["results_csv"] = str(results_csv)

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train YOLOv8 on DeepPCB")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_PROCESSED_DIR / "data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained checkpoint (yolov8n.pt or yolov8s.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--name", default="deeppcb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N backbone layers (for fine-tuning an existing checkpoint)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate override (for fine-tuning, typically lower than a fresh run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker process count (lower this if you hit transient "
        "FileNotFoundError from DataLoader workers on Windows; Ultralytics default is 8)",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=None,
        help="Random translation augmentation fraction (Ultralytics default 0.1). "
        "Raise this to expose training to more boundary-adjacent/partially-cropped "
        "objects, which the default setting under-represents.",
    )
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        run_name=args.name,
        seed=args.seed,
        freeze=args.freeze,
        lr0=args.lr0,
        workers=args.workers,
        translate=args.translate,
    )


if __name__ == "__main__":
    main()
