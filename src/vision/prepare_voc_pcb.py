"""
Convert VOC_PCB (real-photo PCB defect dataset) into the YOLO fine-tuning set.

VOC_PCB.zip is the PKU "PCB-DATASET" in Pascal VOC layout: real camera photos
of bare boards under 12 lighting conditions, annotated with the same 6 defect
concepts as DeepPCB under different names (see VOC_PCB_CLASS_NAME_MAP).

Two things make naive use of this dataset wasteful or wrong, both handled here:

1. 8,001 of the 10,668 images are pure geometric transforms (flip/rotation) of
   the other 2,667 "base" images — same board, same lighting, zero new visual
   information, and redundant with Ultralytics' own live flip/rotation
   augmentation. We only extract and use the 2,667 base images.

2. The dataset's own train/val/test split leaks: rotated/flipped copies of the
   same physical crop can land in different splits. We instead group by the
   crop's stable identity (the filename with the "light_NN_" prefix removed)
   and split at the GROUP level, so every lighting variant of one crop stays
   in a single split.

A small stratified sample of the existing DeepPCB train split is mixed into
the fine-tuning train set as a "replay buffer" — cheap insurance against the
fine-tuned model forgetting the synthetic-image domain.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.vision.constants import (
    CLASS_NAMES,
    DEFAULT_FINETUNE_PROCESSED_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_VOC_RAW_DIR,
    DEFAULT_VOC_ZIP,
    VOC_PCB_CLASS_NAME_MAP,
)

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DEFAULT_REPLAY_COUNT = 400

# Base (non-augmented) files look like "light_04_missing_hole_01_1_600.{xml,jpg}".
# Augmented copies are prefixed "l_" (flip) or "rotation_{90,180,270}_" and are skipped.
_AUGMENTED_PREFIX_RE = re.compile(r"^(l_|rotation_(90|180|270)_)")
_LIGHT_PREFIX_RE = re.compile(r"^light_\d+_")


@dataclass(frozen=True)
class VocObject:
    class_name: str  # already mapped to CLASS_NAMES vocabulary
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (pixels)


@dataclass(frozen=True)
class VocSample:
    image_path: Path
    group_id: str  # crop identity shared across lighting variants
    width: int
    height: int
    objects: tuple[VocObject, ...]


def extract_base_images(zip_path: Path, raw_dir: Path, force: bool = False) -> None:
    """Extract only non-augmented VOC_PCB images+annotations from the zip."""
    if raw_dir.exists():
        if not force:
            logger.info("%s already exists — skipping extraction (use --force to redo)", raw_dir)
            return
        shutil.rmtree(raw_dir)

    if not zip_path.is_file():
        raise FileNotFoundError(f"VOC_PCB zip not found at {zip_path}")

    ann_dir = raw_dir / "Annotations"
    img_dir = raw_dir / "JPEGImages"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if "/Annotations/" not in name and "/JPEGImages/" not in name:
                continue
            stem = Path(name).stem
            if not stem or _AUGMENTED_PREFIX_RE.match(stem):
                continue
            if name.endswith(".xml"):
                dest = ann_dir / f"{stem}.xml"
            elif name.endswith(".jpg"):
                dest = img_dir / f"{stem}.jpg"
            else:
                continue
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1

    logger.info("Extracted %d base files (annotations+images) to %s", extracted, raw_dir)


def _group_id(stem: str) -> str:
    """Strip the 'light_NN_' prefix to get the stable identity of a physical crop."""
    return _LIGHT_PREFIX_RE.sub("", stem)


def _parse_voc_xml(xml_path: Path) -> tuple[int, int, list[VocObject]]:
    """Parse one VOC annotation; unmapped class names are skipped with a debug log."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width", "600"))
    height = int(size.findtext("height", "600"))

    objects: list[VocObject] = []
    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        mapped = VOC_PCB_CLASS_NAME_MAP.get(raw_name)
        if mapped is None:
            logger.debug("Unmapped VOC class '%s' in %s — skipping object", raw_name, xml_path.name)
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        objects.append(VocObject(class_name=mapped, bbox=(xmin, ymin, xmax, ymax)))
    return width, height, objects


def discover_voc_samples(raw_dir: Path) -> list[VocSample]:
    """Parse every base annotation under raw_dir into a VocSample with mapped classes."""
    ann_dir = raw_dir / "Annotations"
    img_dir = raw_dir / "JPEGImages"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"{ann_dir} not found. Run extract_base_images() first.")

    samples: list[VocSample] = []
    skipped_no_image = 0
    skipped_no_objects = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        stem = xml_path.stem
        image_path = img_dir / f"{stem}.jpg"
        if not image_path.is_file():
            skipped_no_image += 1
            continue
        width, height, objects = _parse_voc_xml(xml_path)
        if not objects:
            skipped_no_objects += 1
            continue
        samples.append(
            VocSample(
                image_path=image_path,
                group_id=_group_id(stem),
                width=width,
                height=height,
                objects=tuple(objects),
            )
        )

    if skipped_no_image:
        logger.warning("%d annotations skipped — matching image not found", skipped_no_image)
    if skipped_no_objects:
        logger.warning("%d annotations skipped — no mappable defect objects", skipped_no_objects)
    logger.info("Discovered %d usable VOC_PCB samples across %d groups", len(samples), len({s.group_id for s in samples}))
    return samples


def _group_primary_class(group_samples: list[VocSample]) -> str:
    """Stratify each group by its most frequent defect class across all lighting variants."""
    counts: dict[str, int] = defaultdict(int)
    for sample in group_samples:
        for obj in sample.objects:
            counts[obj.class_name] += 1
    return max(counts, key=lambda k: counts[k])


def grouped_stratified_split(
    samples: list[VocSample],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> dict[str, list[VocSample]]:
    """Split by crop-group (not individual image) so lighting variants never cross splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[str, list[VocSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for group_id, group_samples in groups.items():
        by_class[_group_primary_class(group_samples)].append(group_id)

    split_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for class_name, group_ids in by_class.items():
        rng.shuffle(group_ids)
        n = len(group_ids)
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
        split_groups["train"].extend(group_ids[:n_train])
        split_groups["val"].extend(group_ids[n_train : n_train + n_val])
        split_groups["test"].extend(group_ids[n_train + n_val : n_train + n_val + n_test])
        if n_val == 0:
            logger.warning("Class '%s' has only %d group(s) — 0 assigned to val split", class_name, n)
        if n_test == 0:
            logger.warning("Class '%s' has only %d group(s) — 0 assigned to test split", class_name, n)

    splits: dict[str, list[VocSample]] = {"train": [], "val": [], "test": []}
    for split_name, group_ids in split_groups.items():
        for group_id in group_ids:
            splits[split_name].extend(groups[group_id])
        rng.shuffle(splits[split_name])

    total_out = sum(len(v) for v in splits.values())
    assert total_out == len(samples), (
        f"Split total {total_out} != input count {len(samples)} — logic error in grouped_stratified_split"
    )
    return splits


def _voc_sample_to_yolo_lines(sample: VocSample) -> list[str]:
    lines = []
    for obj in sample.objects:
        xmin, ymin, xmax, ymax = obj.bbox
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        x_center = min(max(((xmin + xmax) / 2.0) / sample.width, 0.0), 1.0)
        y_center = min(max(((ymin + ymax) / 2.0) / sample.height, 0.0), 1.0)
        w = min(max((xmax - xmin) / sample.width, 0.0), 1.0)
        h = min(max((ymax - ymin) / sample.height, 0.0), 1.0)
        class_idx = CLASS_NAMES.index(obj.class_name)
        lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines


def _sample_replay_pairs(processed_dir: Path, n: int, seed: int) -> list[tuple[Path, Path]]:
    """Pick a random stratified-by-nothing (uniform) sample of existing DeepPCB train pairs."""
    img_dir = processed_dir / "images" / "train"
    lbl_dir = processed_dir / "labels" / "train"
    if not img_dir.is_dir():
        logger.warning("%s not found — skipping DeepPCB replay buffer", img_dir)
        return []
    all_images = sorted(img_dir.glob("*.jpg"))
    pairs = [(img, lbl_dir / f"{img.stem}.txt") for img in all_images]
    pairs = [(img, lbl) for img, lbl in pairs if lbl.is_file()]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    chosen = pairs[: min(n, len(pairs))]
    logger.info("Sampled %d/%d DeepPCB train images as replay buffer", len(chosen), len(pairs))
    return chosen


def write_finetune_dataset(
    splits: dict[str, list[VocSample]],
    replay_pairs: list[tuple[Path, Path]],
    output_dir: Path,
    force: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists. Pass --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for split_name, split_samples in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for idx, sample in enumerate(split_samples):
            dest_stem = f"voc_{split_name}_{idx:05d}"
            shutil.copy2(sample.image_path, img_dir / f"{dest_stem}.jpg")
            yolo_lines = _voc_sample_to_yolo_lines(sample)
            (lbl_dir / f"{dest_stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
            )

    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    for idx, (img_path, lbl_path) in enumerate(replay_pairs):
        dest_stem = f"replay_{idx:05d}"
        shutil.copy2(img_path, train_img_dir / f"{dest_stem}.jpg")
        shutil.copy2(lbl_path, train_lbl_dir / f"{dest_stem}.txt")

    data_yaml = output_dir / "data.yaml"
    yaml_content = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    data_yaml.write_text(yaml.safe_dump(yaml_content, sort_keys=False))
    return data_yaml


def prepare_voc_finetune_dataset(
    zip_path: Path = DEFAULT_VOC_ZIP,
    raw_dir: Path = DEFAULT_VOC_RAW_DIR,
    output_dir: Path = DEFAULT_FINETUNE_PROCESSED_DIR,
    deeppcb_processed_dir: Path = DEFAULT_PROCESSED_DIR,
    replay_count: int = DEFAULT_REPLAY_COUNT,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Full pipeline: extract, discover, grouped-split, mix in replay buffer, write YOLO dataset."""
    extract_base_images(zip_path, raw_dir, force=force)
    samples = discover_voc_samples(raw_dir)
    if not samples:
        raise RuntimeError(f"No usable VOC_PCB samples found under {raw_dir}")

    splits = grouped_stratified_split(samples, seed=seed)
    print("VOC_PCB split (image count):")
    for name, split_samples in splits.items():
        print(f"  {name}: {len(split_samples)}")

    replay_pairs = _sample_replay_pairs(deeppcb_processed_dir, replay_count, seed=seed)
    data_yaml = write_finetune_dataset(splits, replay_pairs, output_dir, force=force)
    print(f"Replay buffer (DeepPCB images mixed into train): {len(replay_pairs)}")
    print(f"Wrote fine-tuning YOLO dataset to {output_dir}")
    print(f"Config: {data_yaml}")
    return data_yaml


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare VOC_PCB for fine-tuning")
    parser.add_argument("--zip", type=Path, default=DEFAULT_VOC_ZIP)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_VOC_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINETUNE_PROCESSED_DIR)
    parser.add_argument("--deeppcb-processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--replay-count", type=int, default=DEFAULT_REPLAY_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_voc_finetune_dataset(
        zip_path=args.zip,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        deeppcb_processed_dir=args.deeppcb_processed_dir,
        replay_count=args.replay_count,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
