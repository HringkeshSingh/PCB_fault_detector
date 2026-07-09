"""Confidence threshold sweep — find the optimal per-class threshold for recall.

Runs the current YOLO model on the validation (or test) split at a range of
confidence thresholds and prints per-class precision / recall / F1 at each
level.  Use this to empirically tune PER_CLASS_CONF_THRESHOLD in constants.py.

Usage:
    python scripts/sweep_confidence.py
    python scripts/sweep_confidence.py --data data/processed_finetune/data.yaml --split test
    python scripts/sweep_confidence.py --model models/best_deeppcb_only.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.constants import CLASS_NAMES, DEFAULT_BEST_WEIGHTS, DEFAULT_PROCESSED_DIR

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def sweep(
    model_path: Path,
    data_yaml: Path,
    split: str,
    thresholds: list[float],
) -> list[dict]:
    """Run validation at each threshold, collecting per-class metrics."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    all_results = []

    for conf in thresholds:
        print(f"\n{'=' * 60}")
        print(f"  Threshold: {conf:.2f}")
        print(f"{'=' * 60}")

        val = model.val(data=str(data_yaml), split=split, conf=conf, verbose=False)
        box = getattr(val, "box", None)
        if box is None:
            print("  [ERROR] Validation returned no box metrics — skipping")
            continue

        entry: dict = {
            "threshold": conf,
            "overall": {
                "precision": float(getattr(box, "mp", 0.0) or 0.0),
                "recall": float(getattr(box, "mr", 0.0) or 0.0),
                "map50": float(getattr(box, "map50", 0.0) or 0.0),
            },
            "per_class": {},
        }

        # Compute overall F1
        p, r = entry["overall"]["precision"], entry["overall"]["recall"]
        entry["overall"]["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_p = getattr(box, "p", None)
        per_r = getattr(box, "r", None)
        per_ap50 = getattr(box, "ap50", None)

        for idx, name in enumerate(CLASS_NAMES):
            cp = float(per_p[idx]) if per_p is not None and idx < len(per_p) else 0.0
            cr = float(per_r[idx]) if per_r is not None and idx < len(per_r) else 0.0
            ca = float(per_ap50[idx]) if per_ap50 is not None and idx < len(per_ap50) else 0.0
            cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
            entry["per_class"][name] = {
                "precision": round(cp, 4),
                "recall": round(cr, 4),
                "f1": round(cf1, 4),
                "ap50": round(ca, 4),
            }

        all_results.append(entry)

        # Print table
        print(f"\n  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP50':>8}")
        print(f"  {'-' * 44}")
        for name in CLASS_NAMES:
            m = entry["per_class"][name]
            flag = " ⚠" if m["recall"] < 0.50 else ""
            print(f"  {name:<12} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['ap50']:>8.3f}{flag}")
        print(f"  {'-' * 44}")
        o = entry["overall"]
        print(f"  {'OVERALL':<12} {o['precision']:>8.3f} {o['recall']:>8.3f} {o['f1']:>8.3f} {o['map50']:>8.3f}")

    return all_results


def find_optimal_thresholds(results: list[dict]) -> dict[str, dict]:
    """For each class, find the threshold that maximises F1."""
    best: dict[str, dict] = {}
    for name in CLASS_NAMES:
        best_f1 = -1.0
        best_entry = None
        for entry in results:
            f1 = entry["per_class"][name]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_entry = {
                    "threshold": entry["threshold"],
                    **entry["per_class"][name],
                }
        if best_entry:
            best[name] = best_entry
    return best


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds per class")
    parser.add_argument("--model", type=Path, default=DEFAULT_BEST_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_DIR / "data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Confidence thresholds to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save full results",
    )
    args = parser.parse_args()

    results = sweep(args.model, args.data, args.split, args.thresholds)

    if not results:
        print("\n[ERROR] No valid results collected. Check model/data paths.")
        return

    # Print optimal thresholds
    optimal = find_optimal_thresholds(results)
    print(f"\n{'=' * 60}")
    print("  OPTIMAL THRESHOLDS (maximising F1 per class)")
    print(f"{'=' * 60}")
    print(f"\n  {'Class':<12} {'Threshold':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-' * 46}")
    for name in CLASS_NAMES:
        o = optimal[name]
        print(f"  {name:<12} {o['threshold']:>10.2f} {o['precision']:>8.3f} {o['recall']:>8.3f} {o['f1']:>8.3f}")

    # Print suggested constants.py snippet
    print(f"\n  Suggested PER_CLASS_CONF_THRESHOLD:")
    print(f"  {{")
    for name in CLASS_NAMES:
        print(f'      "{name}": {optimal[name]["threshold"]:.2f},')
    print(f"  }}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\n  Full results saved to {args.output}")


if __name__ == "__main__":
    main()
