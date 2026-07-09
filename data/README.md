# Data Directory

Place raw DeepPCB files here after download.

## Expected layout (VERIFY AFTER DOWNLOAD)

See [DEEPPCB_ASSUMPTIONS.md](../DEEPPCB_ASSUMPTIONS.md) for the full assumptions document.

**Quick reference — we expect:**

```
data/raw/DeepPCB/
└── PCBData/
    └── group{NNNNN}/          # e.g. group00041, group20085
        └── {NNNNN}/           # e.g. 00041, 20085
            ├── {id}_test.jpg  # defective tested image (640×640)
            ├── {id}_temp.jpg  # defect-free template (640×640)
            └── {id}.txt       # bounding-box annotations
```

**Annotation format** (one defect per line in `{id}.txt`):

```
x1,y1,x2,y2,type
```

- `(x1,y1)` = top-left corner, `(x2,y2)` = bottom-right corner (pixel coords)
- `type` = integer class ID (1–6; 0 = background, unused)

| ID | Class name   |
|----|--------------|
| 1  | open         |
| 2  | short        |
| 3  | mousebite    |
| 4  | spur         |
| 5  | copper       |
| 6  | pin-hole     |

## Processed output (created by `src/vision/prepare_data.py`)

```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```
