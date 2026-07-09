"""One-off audit runner — reports findings to stdout (not part of production code)."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import HistoricalCaseCollection
from src.vision.constants import (
    DEEPPCB_CLASS_ID_TO_NAME,
    DEEPPCB_ID_TO_YOLO_IDX,
    DEFAULT_BEST_WEIGHTS,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
)
from src.vision.prepare_data import (
    discover_samples,
    stratified_split,
    convert_annotation_file,
    deeppcb_line_to_yolo,
)


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def audit_raw_data() -> None:
    section("DATA PREP — Raw DeepPCB on disk")
    raw = DEFAULT_RAW_DIR
    if not raw.is_dir():
        print(f"BLOCKER: Raw dir missing: {raw}")
        print("  Cannot verify annotation parsing, splits, or bbox overlays.")
        return

    samples = discover_samples(raw)
    print(f"Discovered {len(samples)} samples")

    # Manual annotation spot-check
    for s in random.Random(42).sample(samples, min(5, len(samples))):
        lines = s.annotation_path.read_text(encoding="utf-8").strip().splitlines()[:3]
        print(f"\n  File: {s.annotation_path.name}")
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 5:
                cid = int(parts[4])
                yolo_idx = DEEPPCB_ID_TO_YOLO_IDX.get(cid, "?")
                name = DEEPPCB_CLASS_ID_TO_NAME.get(cid, "?")
                yolo = deeppcb_line_to_yolo(line)
                print(f"    raw type={cid} ({name}) -> yolo_idx={yolo_idx} | {yolo}")

    # Class ID distribution in raw annotations
    all_ids: list[int] = []
    for s in samples:
        all_ids.extend(s.class_ids)
    print("\n  Raw annotation class counts (DeepPCB IDs):")
    for cid, cnt in sorted(Counter(all_ids).items()):
        print(f"    {cid} ({DEEPPCB_CLASS_ID_TO_NAME[cid]}): {cnt}")

    # Split reproducibility & overlap
    s1 = stratified_split(samples, seed=42)
    s2 = stratified_split(samples, seed=42)
    s3 = stratified_split(samples, seed=99)
    same = all(
        [x.image_path for x in s1[k]] == [x.image_path for x in s2[k]] for k in s1
    )
    diff = any(
        [x.image_path for x in s1[k]] != [x.image_path for x in s3[k]] for k in s1
    )
    print(f"\n  Split reproducible (seed=42): {same}")
    print(f"  Split differs with seed=99: {diff}")

    paths1 = {s.image_path for s in s1["train"]} | {s.image_path for s in s1["val"]} | {s.image_path for s in s1["test"]}
    print(f"  Total unique in splits: {len(paths1)} (source: {len(samples)})")
    overlap = len({s.image_path for s in s1["train"]} & {s.image_path for s in s1["val"]})
    overlap |= len({s.image_path for s in s1["train"]} & {s.image_path for s in s1["test"]})
    print(f"  Cross-split image overlap count: {overlap}")

    # Class balance per split (by primary class AND by any class present)
    print("\n  Primary-class images per split:")
    for split_name, split_samples in s1.items():
        keys = Counter(
            max(Counter(s.class_ids), key=lambda k: Counter(s.class_ids)[k])
            for s in split_samples
        )
        print(f"    {split_name}: {dict(keys)}")

    print("\n  Defect-class INSTANCE counts per split (any class in image):")
    for split_name, split_samples in s1.items():
        inst = Counter()
        for s in split_samples:
            inst.update(s.class_ids)
        print(f"    {split_name}: {{{', '.join(f'{DEEPPCB_CLASS_ID_TO_NAME[k]}:{v}' for k,v in sorted(inst.items()))}}}")

    # Check val/test for missing classes (by instance)
    all_classes = set(DEEPPCB_CLASS_ID_TO_NAME.keys())
    for split in ("val", "test"):
        present = set()
        for s in s1[split]:
            present.update(s.class_ids)
        missing = all_classes - present
        if missing:
            print(f"  WARNING val/test missing class IDs in instances: {missing}")

    # YOLO conversion uses fixed 640 — check actual image sizes
    from PIL import Image
    sizes = Counter()
    for s in random.Random(0).sample(samples, min(20, len(samples))):
        with Image.open(s.image_path) as im:
            sizes[im.size] += 1
    print(f"\n  Sampled image sizes: {dict(sizes)}")
    if sizes and not all(s == (640, 640) for s in sizes):
        print("  WARNING: Not all images are 640x640 — fixed-size normalization may be wrong")


def audit_training() -> None:
    section("TRAINING — artifacts")
    data_yaml = DEFAULT_PROCESSED_DIR / "data.yaml"
    results = PROJECT_ROOT / "models" / "training_results.json"
    best = DEFAULT_BEST_WEIGHTS

    if not data_yaml.is_file():
        print(f"BLOCKER: {data_yaml} missing — prepare_data not run or output deleted")
    else:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text())
        print(f"data.yaml path field: {cfg.get('path')}")
        p = Path(cfg["path"])
        for split in ("train", "val", "test"):
            d = p / cfg[split]
            exists = d.is_dir()
            n = len(list(d.glob("*.jpg"))) if exists else 0
            print(f"  {split}: {d} exists={exists} images={n}")

    if not best.is_file():
        print(f"BLOCKER: {best} missing — training not completed in this workspace")
    else:
        print(f"best.pt size: {best.stat().st_size} bytes")

    if not results.is_file():
        print(f"BLOCKER: {results} missing — no metrics to report")
    else:
        m = json.loads(results.read_text())
        print(json.dumps(m, indent=2))


def audit_cases() -> None:
    section("SYNTHETIC CASES — duplication & plausibility")
    path = PROJECT_ROOT / "data" / "cases" / "historical_cases.json"
    data = json.loads(path.read_text())
    coll = HistoricalCaseCollection.model_validate(data)
    cases = coll.cases
    print(f"Total cases: {len(cases)}")

    # Exact duplicate root_cause texts
    rc = Counter(c.root_cause for c in cases)
    dupes = {k: v for k, v in rc.items() if v > 1}
    print(f"Duplicate root_cause strings: {len(dupes)} unique texts reused")
    for text, n in sorted(dupes.items(), key=lambda x: -x[1])[:5]:
        print(f"  x{n}: {text[:70]}...")

    # Embedding similarity
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [coll.to_embedding_text(c) for c in cases]
    emb = model.encode(texts, normalize_embeddings=True)
    high_pairs = []
    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            sim = float(np.dot(emb[i], emb[j]))
            if sim >= 0.95:
                high_pairs.append((sim, cases[i].case_id, cases[j].case_id))
    print(f"\nPairs with cosine similarity >= 0.95: {len(high_pairs)}")
    for sim, a, b in sorted(high_pairs, reverse=True)[:10]:
        print(f"  {sim:.4f}  {a} <-> {b}")


def audit_chroma() -> None:
    section("CHROMADB — count & metadata")
    import chromadb
    from src.data.ingest_cases import COLLECTION_NAME, DEFAULT_CHROMA_DIR

    json_path = PROJECT_ROOT / "data" / "cases" / "historical_cases.json"
    n_json = len(json.loads(json_path.read_text())["cases"])

    client = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"BLOCKER: collection missing: {e}")
        return

    n_chroma = col.count()
    print(f"JSON cases: {n_json}  |  Chroma vectors: {n_chroma}")
    if n_json != n_chroma:
        print("  BLOCKER: count mismatch")

    sample = col.get(limit=3, include=["metadatas"])
    required = {"case_id", "defect_class", "root_cause", "corrective_action", "severity", "date_recorded", "outcome_notes", "component_type"}
    for meta in sample["metadatas"]:
        missing = required - set(meta.keys())
        print(f"  Sample {meta.get('case_id')}: missing fields={missing or 'none'}")

    # Phrasing robustness
    from src.data.test_retrieval import query_cases
    queries = [
        ("open", "trace discontinuity from etch process"),
        ("open", "broken copper connection thermal cycle under-etch"),
        ("short", "unwanted bridge between pads contamination"),
        ("short", "solder bridging adjacent lands stencil issue"),
        ("pin-hole", "void in plane plating bubble"),
        ("pin-hole", "missing copper fill electroless seed failure"),
    ]
    print("\n  Phrasing robustness (top-1 class):")
    for expected, q in queries:
        hits = query_cases(q, n_results=1)
        got = hits[0]["defect_class"] if hits else None
        ok = got == expected
        print(f"    [{ 'OK' if ok else 'FAIL'}] expect={expected} got={got} | {q[:50]}")


if __name__ == "__main__":
    random.seed(42)
    audit_raw_data()
    audit_training()
    audit_cases()
    audit_chroma()
