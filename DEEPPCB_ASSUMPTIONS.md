# DeepPCB Dataset — Assumptions for `prepare_data.py`

> **Verified 2026-07-03** against a real clone of https://github.com/tangsanli5201/DeepPCB.
> The actual layout differs from the original assumptions below — see "Actual layout" note
> in section 2 and the annotation format correction in section 4.

Sources: [DeepPCB README](https://github.com/tangsanli5201/DeepPCB), community usage
(e.g. Detectron2 examples referencing `PCBData/group20085/20085/20085000_test.jpg`).

---

## 1. Root path

We assume you place the extracted dataset at:

```
data/raw/DeepPCB/PCBData/
```

If your archive extracts differently (e.g. flat `PCBData/` at repo root, or nested
`DeepPCB-master/PCBData/`), update the `--raw-dir` argument in `prepare_data.py`
(or set `DEEPPCB_RAW_DIR` env var once that script exists).

---

## 2. Directory hierarchy

**Assumption A — nested group folders (most common in published repos):**

```
PCBData/
├── group00041/
│   └── 00041/
│       ├── 00041000_test.jpg
│       ├── 00041000_temp.jpg
│       ├── 00041000.txt
│       ├── 00041001_test.jpg
│       └── ...
├── group20085/
│   └── 20085/
│       ├── 20085000_test.jpg
│       └── ...
└── ...
```

**Assumption B — alternative flat layout (some forks reorganize):**

```
PCBData/
├── images/       # all *_test.jpg
├── labels/       # all *.txt
└── templates/    # all *_temp.jpg  (optional)
```

`prepare_data.py` will be written for **Assumption A** first. If your download matches
**B**, tell us and we will adapt the discovery logic.

**Actual layout (verified):** neither A nor B exactly — it's a variant of A where
annotations live in a *sibling* `{id}_not/` folder, not inside `{id}/` alongside the images:

```
PCBData/
├── group92000/
│   ├── 92000/
│   │   ├── 92000000_test.jpg
│   │   ├── 92000000_temp.jpg
│   │   └── ...
│   └── 92000_not/
│       ├── 92000000.txt
│       └── ...
└── ...
```

`discover_samples()` in `prepare_data.py` checks this path as a third candidate
alongside Assumption A's two.

---

## 3. File naming

| Suffix / pattern | Role                                      |
|------------------|-------------------------------------------|
| `*_test.jpg`     | Defective image — **used for training**   |
| `*_temp.jpg`     | Template pair — **ignored in Phase 1**    |
| `{same_id}.txt`  | Annotations for the `*_test.jpg` image    |

- Image size: **640 × 640** pixels (per README).
- Image count: ~1,500 pairs total; official split is 1,000 train / 500 test.
  Our script will re-split 80/10/10 from all annotated images unless you prefer
  the official split (configurable).

---

## 4. Annotation file format

One line per defect. Originally assumed comma-separated; **actual files are
space-separated**:

```
x1 y1 x2 y2 type
```

Example (real file, `group92000/92000_not/92000000.txt`):

```
162 397 243 436 1
296 102 377 141 2
308 334 360 415 3
```

`_read_class_ids()` / `deeppcb_line_to_yolo()` in `prepare_data.py` accept either
comma- or space-separated lines (normalizes commas to spaces before splitting).

- Coordinates are absolute pixels in the 640×640 image.
- Class IDs 1–6 map to YOLO class indices 0–5:

| DeepPCB ID | YOLO idx | Name       |
|------------|----------|------------|
| 1          | 0        | open       |
| 2          | 1        | short      |
| 3          | 2        | mousebite  |
| 4          | 3        | spur       |
| 5          | 4        | copper     |
| 6          | 5        | pin-hole   |

---

## 5. Which images to train on

**Phase 1 trains on `*_test.jpg` only** (the defective image with ground-truth boxes).
Template images (`*_temp.jpg`) are excluded unless you later want a difference-image
pipeline.

---

## 6. Things to verify on your machine

After download, please check:

1. [ ] Does `data/raw/DeepPCB/PCBData/` exist (or tell us the actual path)?
2. [ ] Are files under `group*/` subfolders (Assumption A) or a flat layout (B)?
3. [ ] Do `.txt` files share the same stem as `*_test.jpg` (e.g. `20085000.txt` ↔ `20085000_test.jpg`)?
4. [ ] Open one `.txt` — does each line match `x1,y1,x2,y2,type` with integer coords?
5. [ ] Are images `.jpg` (not `.png` or `.bmp`)?

Reply with your findings and we will proceed to `prepare_data.py`.
