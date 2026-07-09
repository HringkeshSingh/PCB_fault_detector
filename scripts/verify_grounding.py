"""
Phase 4 grounding verification.

Runs the full pipeline (detect -> retrieve -> score -> generate) on a test image and
checks whether the generated report is actually grounded in the retrieved context —
i.e. that the LLM did not invent case_ids, root causes, or contradict an empty
retrieval. A failing grounding score means the SYSTEM_PROMPT needs adjusting, not the code.

Run:
    python scripts/verify_grounding.py --image path/to/pcb.jpg

Notes:
- Requires a trained model (models/best.pt) and a populated chroma_db/.
- Grounding is only meaningfully tested when Ollama is running (generation_mode="llm").
  In fallback mode every report is grounded-by-construction (unsupported=True,
  evidence_basis=[]) — the script says so explicitly rather than reporting a hollow 100%.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.api.main import _report_generator, _run_retrieval_pipeline, get_detector
from src.retrieval.schemas import CaseResult

# Default test image: the real-photo missing-hole sample if present, else a DeepPCB test image.
_DEFAULT_CANDIDATES = [
    PROJECT_ROOT / "l_light_01_missing_hole_01_2_600.jpg",
    PROJECT_ROOT / "data" / "processed_finetune" / "images" / "test",
    PROJECT_ROOT / "data" / "processed" / "images" / "test",
]

_WORD_RE = re.compile(r"[a-z]{5,}")  # significant words (>=5 chars) for the vocabulary check


def _pick_default_image() -> Path | None:
    """Choose a sensible default test image if the user didn't pass one."""
    for cand in _DEFAULT_CANDIDATES:
        if cand.is_file():
            return cand
        if cand.is_dir():
            jpgs = sorted(cand.glob("*.jpg"))
            if jpgs:
                return jpgs[0]
    return None


def _vocab_overlap(primary_cause: str, cases: list[CaseResult]) -> bool:
    """True if primary_cause shares a significant word with any retrieved case's root_cause."""
    pc_words = set(_WORD_RE.findall(primary_cause.lower()))
    for case in cases:
        if pc_words & set(_WORD_RE.findall(case.root_cause.lower())):
            return True
    return False


def main() -> int:
    """Run the pipeline on a test image and print a grounding score. Returns process exit code."""
    parser = argparse.ArgumentParser(description="Verify Phase 4 report grounding")
    parser.add_argument("--image", type=Path, default=None, help="Test image path")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    image_path = args.image or _pick_default_image()
    if image_path is None or not Path(image_path).is_file():
        print(f"ERROR: no test image found (looked for {image_path}). Pass --image.")
        return 2

    detector = get_detector(conf_threshold=args.conf, iou_threshold=args.iou)
    if detector is None:
        print("ERROR: model not loaded (models/best.pt missing). Train first.")
        return 2

    print(f"Image: {image_path}")
    image = Image.open(io.BytesIO(Path(image_path).read_bytes())).convert("RGB")
    detection_result = detector.detect(image)
    try:
        analysis = _run_retrieval_pipeline(detection_result)
    except Exception as exc:
        print(f"ERROR: retrieval pipeline failed ({exc}). Is chroma_db/ populated?")
        return 2

    report = _report_generator.generate_report(
        analysis,
        image_width=detection_result.image_width,
        image_height=detection_result.image_height,
    )

    mode = report.generation_metadata.generation_mode
    print(f"Generation mode: {mode}  (LLM available: {_report_generator.llm_available})")
    if mode == "fallback":
        print(
            "NOTE: running in fallback mode — reports are grounded by construction "
            "(unsupported=True, evidence_basis=[]). Start Ollama to test LLM grounding."
        )
    print(f"Defects: {report.total_defects}\n")

    fully_grounded = 0
    total = len(report.defect_reports)

    # defect_reports are produced in the same order as analysis.results.
    for result, dr in zip(analysis.results, report.defect_reports):
        retrieved_ids = {c.case_id for c in result.retrieved_cases}
        issues: list[str] = []

        # (a) every cited case_id must exist in the retrieved cases.
        invented = [cid for cid in dr.root_cause.evidence_basis if cid not in retrieved_ids]
        if invented:
            issues.append(f"cites case_ids not retrieved: {invented}")

        # (b) primary_cause should share vocabulary with a retrieved case (LLM mode only —
        #     fallback text is intentionally generic and unsupported).
        if dr.generated_by == "llm" and result.retrieved_cases and not _vocab_overlap(
            dr.root_cause.primary_cause, result.retrieved_cases
        ):
            issues.append("primary_cause shares no vocabulary with any retrieved case")

        # (c) contradiction: claims support but nothing was retrieved.
        if not dr.root_cause.unsupported and not result.retrieved_cases:
            issues.append("unsupported=False but retrieved_cases was empty")

        status = "GROUNDED" if not issues else "FLAGGED"
        if not issues:
            fully_grounded += 1
        print(f"[{status}] {dr.defect_class} ({dr.generated_by}) @ {dr.location}")
        print(f"    evidence_basis: {dr.root_cause.evidence_basis or '[]'}  (retrieved: {sorted(retrieved_ids) or '[]'})")
        for issue in issues:
            print(f"    - {issue}")

    print(f"\nGrounding score: {fully_grounded}/{total} defect reports fully grounded")
    return 0 if fully_grounded == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
