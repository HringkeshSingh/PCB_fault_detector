"""
Convert DeepPCB annotations to YOLO format and split train/val/test.

Raw data lives under data/raw/DeepPCB/PCBData/group{NNNNN}/{NNNNN}/ with
images named {id}_test.jpg / {id}_temp.jpg, and annotations in a sibling
{NNNNN}_not/{id}.txt folder (space-separated "x1 y1 x2 y2 type" lines).
See DEEPPCB_ASSUMPTIONS.md at project root.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import yaml

from src.vision.constants import (
    CLASS_NAMES,
    DEEPPCB_CLASS_ID_TO_NAME,
    DEEPPCB_ID_TO_YOLO_IDX,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Warn if more than this fraction of annotation lines are dropped.
DROP_RATE_WARN_THRESHOLD = 0.05


@dataclass(frozen=True)
class AnnotatedSample:
    """One defective test image and its DeepPCB annotation file."""

    image_path: Path
    annotation_path: Path
    class_ids: tuple[int, ...]


class _ReadResult(NamedTuple):
    class_ids: list[int]
    skipped_lines: int
    total_lines: int


def discover_samples(raw_dir: Path) -> list[AnnotatedSample]:
    """
    Discover annotated test images under DeepPCB PCBData.

    Supports Assumption A (nested group folders) and Assumption B (flat images/ + labels/).
    Logs a warning for every image that has no matching annotation file.
    """
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw DeepPCB directory not found: {raw_dir}\n"
            "Expected: data/raw/DeepPCB/PCBData/ — see DEEPPCB_ASSUMPTIONS.md"
        )

    samples: list[AnnotatedSample] = []
    missing_ann = 0

    # Assumption B: flat layout with images/ and labels/
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        all_images = sorted(images_dir.glob("*_test.jpg"))
        for image_path in all_images:
            stem = image_path.name.replace("_test.jpg", "")
            ann_path = labels_dir / f"{stem}.txt"
            if not ann_path.is_file():
                ann_path = labels_dir / f"{stem}_test.txt"
            if not ann_path.is_file():
                logger.warning("No annotation found for %s — skipping", image_path.name)
                missing_ann += 1
                continue
            result = _read_class_ids(ann_path)
            if result.class_ids:
                samples.append(
                    AnnotatedSample(image_path, ann_path, tuple(result.class_ids))
                )
            else:
                logger.warning(
                    "Annotation %s yielded no valid class IDs (%d/%d lines skipped) — skipping image",
                    ann_path.name,
                    result.skipped_lines,
                    result.total_lines,
                )
                missing_ann += 1
        if samples:
            if missing_ann:
                logger.warning(
                    "Flat layout: skipped %d/%d images (missing or empty annotation)",
                    missing_ann,
                    len(all_images),
                )
            return samples

    # Assumption A: nested group{NNNNN}/{NNNNN}/
    all_test_images = sorted(raw_dir.rglob("*_test.jpg"))
    for test_image in all_test_images:
        stem = test_image.name.replace("_test.jpg", "")
        ann_candidates = [
            test_image.with_name(f"{stem}.txt"),
            test_image.parent / f"{stem}.txt",
            # Assumption C: annotations in a sibling `{id}_not/` folder, e.g.
            # PCBData/group92000/92000/92000000_test.jpg ->
            # PCBData/group92000/92000_not/92000000.txt
            test_image.parent.parent / f"{test_image.parent.name}_not" / f"{stem}.txt",
        ]
        ann_path = next((p for p in ann_candidates if p.is_file()), None)
        if ann_path is None:
            logger.warning("No annotation found for %s — skipping", test_image.name)
            missing_ann += 1
            continue
        result = _read_class_ids(ann_path)
        if result.class_ids:
            samples.append(AnnotatedSample(test_image, ann_path, tuple(result.class_ids)))
        else:
            logger.warning(
                "Annotation %s yielded no valid class IDs (%d/%d lines skipped) — skipping image",
                ann_path.name,
                result.skipped_lines,
                result.total_lines,
            )
            missing_ann += 1

    if missing_ann:
        logger.warning(
            "Nested layout: skipped %d/%d images (missing or empty annotation)",
            missing_ann,
            len(all_test_images),
        )

    if not samples:
        raise FileNotFoundError(
            f"No annotated *_test.jpg samples found under {raw_dir}. "
            "Check DEEPPCB_ASSUMPTIONS.md and your download layout."
        )
    return samples


def _read_class_ids(annotation_path: Path) -> _ReadResult:
    """Parse DeepPCB .txt and return valid class IDs (1-6) plus skip/total counts."""
    class_ids: list[int] = []
    skipped = 0
    lines = annotation_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 5:
            logger.debug("Malformed line in %s (expected 5 fields): %r", annotation_path.name, line)
            skipped += 1
            continue
        try:
            class_id = int(parts[4])
        except ValueError:
            logger.debug("Non-integer class ID in %s: %r", annotation_path.name, parts[4])
            skipped += 1
            continue
        if class_id not in DEEPPCB_CLASS_ID_TO_NAME:
            logger.debug("Unknown class ID %d in %s", class_id, annotation_path.name)
            skipped += 1
            continue
        class_ids.append(class_id)
    return _ReadResult(class_ids, skipped, len(lines))


def deeppcb_line_to_yolo(
    line: str,
    img_w: int = IMAGE_WIDTH,
    img_h: int = IMAGE_HEIGHT,
) -> str | None:
    """Convert one DeepPCB line (x1,y1,x2,y2,type) to YOLO normalized line."""
    parts = line.strip().replace(",", " ").split()
    if len(parts) != 5:
        return None
    try:
        x1, y1, x2, y2 = (float(parts[i]) for i in range(4))
        class_id = int(parts[4])
    except ValueError:
        return None
    if class_id not in DEEPPCB_ID_TO_YOLO_IDX:
        return None

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    yolo_idx = DEEPPCB_ID_TO_YOLO_IDX[class_id]
    return f"{yolo_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_annotation_file(
    annotation_path: Path,
    image_path: Path | None = None,
) -> list[str]:
    """Convert a DeepPCB annotation file to YOLO label lines.

    Reads actual image dimensions from *image_path* when provided, falling back
    to the IMAGE_WIDTH/IMAGE_HEIGHT constants. Warns if size differs from expected.
    """
    img_w, img_h = IMAGE_WIDTH, IMAGE_HEIGHT
    if image_path is not None:
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(image_path) as img:
                img_w, img_h = img.size
            if img_w != IMAGE_WIDTH or img_h != IMAGE_HEIGHT:
                logger.warning(
                    "Image %s is %dx%d, expected %dx%d — using actual size for normalisation",
                    image_path.name,
                    img_w,
                    img_h,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                )
        except Exception as exc:
            logger.warning("Could not read dimensions from %s (%s); using defaults", image_path, exc)

    yolo_lines: list[str] = []
    total_lines = 0
    skipped_lines = 0
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total_lines += 1
        converted = deeppcb_line_to_yolo(line, img_w, img_h)
        if converted:
            yolo_lines.append(converted)
        else:
            skipped_lines += 1

    if total_lines > 0:
        drop_rate = skipped_lines / total_lines
        if drop_rate > DROP_RATE_WARN_THRESHOLD:
            logger.warning(
                "%s: dropped %d/%d annotation lines (%.0f%%)",
                annotation_path.name,
                skipped_lines,
                total_lines,
                drop_rate * 100,
            )

    return yolo_lines


def stratify_key(sample: AnnotatedSample) -> str:
    """Primary defect class in the image (most frequent) for stratified splitting."""
    if not sample.class_ids:
        return "none"
    counts = Counter(sample.class_ids)
    primary = counts.most_common(1)[0][0]
    return DEEPPCB_CLASS_ID_TO_NAME[primary]


def stratified_split(
    samples: list[AnnotatedSample],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> dict[str, list[AnnotatedSample]]:
    """Split samples into train/val/test, stratified by primary defect class.

    Warns when any class ends up with zero samples in val or test.
    Asserts that total counts across splits equal the input length.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    by_class: dict[str, list[AnnotatedSample]] = defaultdict(list)
    for sample in samples:
        by_class[stratify_key(sample)].append(sample)

    splits: dict[str, list[AnnotatedSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for class_name, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n - n_train - n_val, 1)
            if n_train + n_val + n_test > n:
                n_train = max(n - n_val - n_test, 1)
        else:
            n_test = n - n_train - n_val

        if n_val == 0:
            logger.warning(
                "Class '%s' has only %d image(s) — 0 assigned to val split; "
                "val metrics will not reflect this class",
                class_name,
                n,
            )
        if n_test == 0:
            logger.warning(
                "Class '%s' has only %d image(s) — 0 assigned to test split",
                class_name,
                n,
            )

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val : n_train + n_val + n_test])

    for key in splits:
        rng.shuffle(splits[key])

    total_out = sum(len(v) for v in splits.values())
    assert total_out == len(samples), (
        f"Split total {total_out} != input count {len(samples)} — logic error in stratified_split"
    )

    return splits


def write_yolo_dataset(
    splits: dict[str, list[AnnotatedSample]],
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Copy images and write YOLO labels; return path to data.yaml.

    Raises RuntimeError if output_dir exists and *force* is False.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists. "
                "Pass force=True (or --force on CLI) to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for split_name, split_samples in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for idx, sample in enumerate(split_samples):
            dest_stem = f"{split_name}_{idx:05d}"
            dest_img = img_dir / f"{dest_stem}.jpg"
            dest_lbl = lbl_dir / f"{dest_stem}.txt"
            shutil.copy2(sample.image_path, dest_img)
            yolo_lines = convert_annotation_file(sample.annotation_path, sample.image_path)
            dest_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    # Ultralytics resolves `path` against its own working directory / settings,
    # not against data.yaml's location — a relative "." silently breaks unless
    # the process cwd happens to match. Use an absolute path instead.
    data_yaml = output_dir / "data.yaml"
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    data_yaml.write_text(yaml.safe_dump(yaml_content, sort_keys=False))
    return data_yaml


def prepare_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Full pipeline: discover, split, convert, write data.yaml."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    samples = discover_samples(raw_dir)
    splits = stratified_split(samples, seed=seed)
    data_yaml = write_yolo_dataset(splits, output_dir, force=force)

    print(f"Discovered {len(samples)} annotated images")
    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)}")
    print(f"Wrote YOLO dataset to {output_dir}")
    print(f"Config: {data_yaml}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepPCB to YOLO format (80/10/10 stratified split)"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Path to DeepPCB PCBData directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for YOLO dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory without prompting",
    )
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
