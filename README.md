# 🔬 PCB Defect Report Generator

> Multimodal AI inspection system — YOLOv8 defect detection fused with a RAG pipeline over historical manufacturing cases and IPC standards, then a local-LLM report generator, delivering a structured, cited inspection report from a single image upload.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-2.2+-yellow)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000)
![Build](https://img.shields.io/badge/Build-Phase%204%20of%206-yellowgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Table of Contents

- [What This Does](#-what-this-does)
- [Quick Start (End-to-End)](#-quick-start-end-to-end)
- [Pretrained Model](#-pretrained-model)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Defect Classes](#-defect-classes)
- [Setup & Installation](#-setup--installation)
- [Real-Photo Fine-Tuning (VOC_PCB)](#-real-photo-fine-tuning-voc_pcb)
- [Report Generation (Phase 4 · Local LLM)](#-report-generation-phase-4--local-llm)
- [Running the API](#-running-the-api)
- [Running the GUI](#-running-the-gui)
- [API Reference](#-api-reference)
- [Configuration Reference](#-configuration-reference)
- [Phase Roadmap](#-phase-roadmap)
- [Development Notes](#-development-notes)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🧭 What This Does

Manual PCB inspection requires a trained engineer to visually identify a defect, recall similar historical cases from memory, cross-reference IPC standards, and write a structured report — a process that takes 10–20 minutes per board and varies by inspector.

This system replaces that loop: upload a PCB image, receive a structured JSON report with detected defect locations, matched historical cases, relevant IPC standards excerpts, a confidence score, a human-review flag, and — when a local LLM is available — a cited, plain-English root-cause narrative and corrective-action plan.

**Four phases, one pipeline:**

| Phase | What it does | Key tech |
|-------|-------------|----------|
| **Phase 1 — Vision** | Detects and localises defects in PCB images | YOLOv8 fine-tuned on DeepPCB + real photos |
| **Phase 2 — Case Database** | Searchable knowledge base of historical defect cases + IPC standards | ChromaDB, sentence-transformers |
| **Phase 3 — RAG Retrieval** | Matches detections to past cases and standards with confidence scoring | Semantic search, Pydantic contracts |
| **Phase 4 — Report Generation** | Turns retrieval output into a cited, confidence-aware inspection report | Local LLM via Ollama, Pydantic-validated, fallback-safe |

---

## 🚀 Quick Start (End-to-End)

This is the full path from a fresh clone to analyzing a board in the GUI. Run every command from the project root (`PCB_faultDetection/`) with your virtual environment activated. Later sections ([Setup & Installation](#-setup--installation), [Running the GUI](#-running-the-gui)) go deeper on each step.

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

**GPU users (recommended for training):** the default `pip install torch` pulls the CPU-only build. On CPU, training is ~90 hours; on a modest NVIDIA GPU it is ~2–3 hours. If you have an NVIDIA card, replace torch with a CUDA build **after** the install above:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Ultralytics auto-detects the GPU — no code change needed. It prints `CUDA:0 (<your GPU>)` at training startup to confirm.

### 2. Get the dataset

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB
```

> The clone is ~230 MB. If it times out mid-download, `cd data/raw/DeepPCB && git fetch && git checkout master` to finish it. The dataset's real layout (annotations in sibling `{id}_not/` folders, space-separated labels) is handled automatically by `prepare_data.py` — see [DEEPPCB_ASSUMPTIONS.md](DEEPPCB_ASSUMPTIONS.md).

### 3. Prepare the data (DeepPCB → YOLO format)

```bash
python -m src.vision.prepare_data
```

Produces `data/processed/` with `images/{train,val,test}/`, matching `labels/`, and `data.yaml`. Expect ~1197 train / 148 val / 155 test.

### 4. Train the detector

```bash
# GPU with limited VRAM (e.g. 4 GB) — use a smaller batch to avoid OOM
python -m src.vision.train --model yolov8s.pt --epochs 50 --batch 8

# Faster / lower-accuracy alternative
python -m src.vision.train --model yolov8n.pt --epochs 50 --batch 16
```

Writes `models/best.pt` (the weights the API loads) and `models/training_results.json` (per-class mAP/precision/recall).

### 5. Build the retrieval database (one time)

```bash
python -m src.data.generate_cases      # -> data/cases/historical_cases.json (180 cases)
python -m src.data.ingest_cases        # -> ChromaDB collection: pcb_defect_cases
python -m src.data.ingest_standards    # -> ChromaDB collection: pcb_standards
```

### 6. Start the backend + GUI

```bash
# Terminal 1 — API (loads best.pt + ChromaDB; first request is slow ~25-30s)
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — GUI
streamlit run src/gui/app.py
```

### 7. Analyze a board

1. Open `http://localhost:8501`.
2. Check the sidebar: it should say **"Model loaded ✓"**. If it says "Model not loaded", step 4 hasn't produced `models/best.pt` yet; if it says "Cannot reach API", the API in Terminal 1 isn't up.
3. Choose **Upload file** (pick a PCB `.jpg`/`.png`) or **Webcam snapshot** (take one frame).
4. Click **Analyze**. You'll get:
   - The image with color-coded bounding boxes (green=high, yellow=medium, orange=low, red=uncertain confidence)
   - Per-defect cards with detection confidence, top case similarity, and a human-review flag when warranted
   - Expandable tabs of matched **historical cases** (root cause, corrective action, outcome) and **IPC standards** excerpts

> **Prefer the API directly?** Skip the GUI and `curl` the endpoints — see [API Reference](#-api-reference). The GUI is just a thin client over `POST /analyze`.

### 8. (Optional) Enable LLM inspection reports

`POST /report` produces a cited, plain-English inspection report. It works **without** an LLM (static fallback templates), but for generated narratives install a local model:

```bash
# Install Ollama from https://ollama.com/download, then in any terminal:
ollama pull llama3.2
```

Restart the API and it auto-detects the model — `GET /report/health` flips to `llm_available: true`. See [Report Generation](#-report-generation-phase-4--local-llm) for details.

---

## 🧠 Pretrained Model

A **pretrained model checkpoint** (`models/best.pt`) is included in this repository, so you can skip training entirely and go straight to inference.

| Property | Value |
|----------|-------|
| **File** | `models/best.pt` (~22 MB) |
| **Architecture** | YOLOv8s |
| **Training** | DeepPCB (50 epochs) → VOC_PCB real-photo fine-tune (50 total epochs across 3 passes) |
| **Real-photo mAP@0.5** | **0.788** (held-out VOC_PCB test set) |
| **Synthetic mAP@0.5** | **0.973** (DeepPCB test set — near-zero forgetting) |
| **Defect classes** | `open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole` |

### Using the pretrained model

After cloning and installing dependencies, you can **skip steps 2–4** (dataset download, data preparation, and training) in the [Quick Start](#-quick-start-end-to-end) and jump directly to step 5 (building the retrieval database) and step 6 (starting the API + GUI):

```bash
# Install dependencies
pip install -r requirements.txt

# Build the retrieval database (still required)
python -m src.data.generate_cases
python -m src.data.ingest_cases
python -m src.data.ingest_standards

# Start the API + GUI
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
streamlit run src/gui/app.py
```

The API automatically loads `models/best.pt` at startup. No additional configuration needed.

> **Want to retrain or fine-tune further?** See [Setup & Installation](#-setup--installation) and [Real-Photo Fine-Tuning](#-real-photo-fine-tuning-voc_pcb) for the full training pipeline. Back up the included `best.pt` first if you want to keep it.

---

## 🏗 Architecture

```
+---------------------------------------------------------------------+
|                         FastAPI REST API                            |
|                      (src/api/main.py)                              |
|                                                                     |
|   GET  /health         -- model load status                        |
|   POST /detect         -- image -> defect list + bounding boxes     |
|   POST /analyze        -- image -> defect list + RAG + confidence   |
|   POST /report         -- image -> full LLM inspection report       |
|   GET  /report/health  -- LLM availability / fallback-mode flag     |
+----------+----------------------------------+------------------------+
           |                                  |
           v                                  v
+---------------------+     +------------------------------------------+
|  Phase 1: Vision    |     |  Phase 3: RAG Retrieval Pipeline         |
|  (src/vision/)      |     |  (src/retrieval/)                        |
|                     |     |                                          |
|  prepare_data.py    |     |  +-------------+   +------------------+  |
|       |             |     |  | QueryBuilder| ->|  PCBRetriever    |  |
|  train.py (YOLOv8)  |     |  |             |   |  retrieve_cases  |  |
|       |             |     |  | Detections  |   |  retrieve_stds   |  |
|  detect.py     -----+---->|  | -> Queries  |   |  combine results |  |
|  (DefectDetector)   |     |  +-------------+   +--------+---------+  |
+---------------------+     |                            |            |
                            |                   +--------v----------+ |
                            |                   | ConfidenceScorer  | |
                            |                   | score + flag      | |
                            |                   +-------------------+ |
                            +-----------------+------------------------+
                                              |
                                              v
                            +----------------------------------+
                            |      ChromaDB Vector Store       |
                            |      (chroma_db/)                |
                            |                                  |
                            |  pcb_defect_cases (180 vectors)  |
                            |  synthetic historical cases      |
                            |                                  |
                            |  pcb_standards (18 vectors)      |
                            |  IPC-A-610 / IPC-6012 excerpts   |
                            +---------------+------------------+
                                            ^
                                            |  Ingestion (run once)
                            +---------------+------------------+
                            |  Phase 2: Case Database          |
                            |  (src/data/)                     |
                            |  generate_cases.py  -> JSON      |
                            |  ingest_cases.py    -> ChromaDB  |
                            |  ingest_standards.py -> ChromaDB |
                            +----------------------------------+

  AnalysisResult (Pydantic)
        |
        v
+------------------------------------------------------------+
|  Phase 4: Report Generation  (src/generation/)             |
|                                                            |
|  ReportGenerator.generate_report(AnalysisResult)           |
|    per defect: LLM (Ollama, grounded prompt) OR fallback   |
|    -> Pydantic-validated DefectReport (never raw text)     |
|    -> InspectionReport (severity, root cause, actions)     |
+------------------------------------------------------------+
```

### Data Flow

1. **Image in** -> `DefectDetector` runs YOLOv8 inference -> list of `Detection` objects `{defect_class, confidence, bbox}`
2. **QueryBuilder** converts each detection into a typed `RetrievalQuery` with confidence banding
3. **PCBRetriever** queries ChromaDB — `defect_class` metadata pre-filter applied before semantic search, one query per defect class
4. **ConfidenceScorer** evaluates detection + retrieval quality, sets `retrieval_confidence` and `flagged_for_human_review`
5. **AnalysisResult** (Pydantic) returned — typed contract consumed by Phase 4
6. **ReportGenerator** (Phase 4) turns each `RetrievalResult` into a Pydantic-validated `DefectReport` — via the local LLM when available and the band is not `uncertain`, else a static fallback template — and assembles the `InspectionReport`

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`$eq` metadata filter before semantic search** | Prevents cross-class contamination — a "mousebite" query never returns "short" cases even if embeddings are close |
| **One ChromaDB query per defect class** | Multi-defect images need separate retrievals; merging degrades top-k quality for each class |
| **Pydantic `AnalysisResult` as Phase 3→4 contract** | Typed, validated interface so Phase 4 LLM generator receives consistent, schema-checked data |
| **Separate `pcb_defect_cases` + `pcb_standards` collections** | Different document structures and query strategies; standards can be absent without blocking case retrieval |
| **Confidence banding (high/medium/low) not raw float** | The LLM in Phase 4 should reason about uncertainty categorically, not do arithmetic on a raw float |
| **YOLOv8 over YOLOv5** | Simpler API, active maintenance, native metrics export, no separate weight download step |
| **Local LLM (Ollama) for Phase 4, not a cloud API** | No API key, fully offline, no per-call cost or data egress; fits a single-workstation deployment |
| **Pydantic validation after every LLM call; fallback on failure** | Raw LLM text never reaches the frontend; a malformed or hallucinated response degrades to a schema-identical fallback, so `/report` always returns a valid report |
| **Grounding enforced in the prompt, verified by a script** | Root causes/citations must come from retrieved cases only; `verify_grounding.py` checks this rather than trusting the model |

---

### Report Generation (Phase 4) at a glance

`POST /report` runs the whole pipeline, then generates a structured `InspectionReport`. Each defect becomes a `DefectReport` with a severity rationale, a **grounded** root-cause analysis (`evidence_basis` = the exact `case_id`s relied on), a corrective-action plan, and a technician-facing narrative. Two modes, schema-identical:

- **LLM mode** (`generated_by="llm"`) — used when Ollama is reachable, the model is pulled, and the retrieval band is not `uncertain`. The prompt passes only the retrieved cases/standards and forbids citing anything outside them.
- **Fallback mode** (`generated_by="fallback"`, `root_cause.unsupported=true`) — used when the LLM is unavailable, a call fails, output fails validation, or the band is `uncertain`. Static per-class templates whose vocabulary is drawn from `historical_cases.json`.

`generate_report()` **never raises** — failures are logged and always resolve to a valid report.

---

## 📁 Project Structure

```
PCB_faultDetection/
|
+-- README.md                      # This file
+-- DEEPPCB_ASSUMPTIONS.md         # Documents assumed DeepPCB folder layout
+-- requirements.txt               # Runtime Python dependencies
+-- requirements-dev.txt           # Test dependencies (pytest)
+-- pytest.ini                     # Pytest marker config (integration tests)
+-- .gitignore                     # Excludes data/, models/, chroma_db/
|
+-- VOC_PCB.zip                    # Real-photo PCB defect dataset (gitignored, see below)
|
+-- data/
|   +-- raw/                       # Place raw DeepPCB dataset here (gitignored)
|   |   +-- DeepPCB/PCBData/       # Expected dataset root
|   |   +-- VOC_PCB/               # Extracted VOC_PCB base images (gitignored)
|   |       +-- Annotations/       # VOC XML, non-augmented files only
|   |       +-- JPEGImages/        # Real camera photos, non-augmented files only
|   +-- processed_finetune/        # YOLO-format real-photo fine-tuning set (gitignored)
|   +-- cases/
|       +-- historical_cases.json  # 180 synthetic historical defect cases
|
+-- src/
|   +-- vision/                    # Phase 1: Computer Vision Pipeline
|   |   +-- constants.py           # Class name->ID mapping, default paths
|   |   +-- prepare_data.py        # DeepPCB -> YOLO format + stratified split
|   |   +-- prepare_voc_pcb.py     # VOC_PCB -> YOLO fine-tuning set (dedup + leak-free split)
|   |   +-- train.py               # YOLOv8 training/fine-tuning, saves best.pt + metrics
|   |   +-- detect.py              # DefectDetector: inference + bbox rescaling
|   |   +-- test_prepare_data.py   # Unit tests for YOLO coordinate normalisation
|   |
|   +-- data/                      # Phase 2: Historical Case Database
|   |   +-- schemas.py             # Pydantic model: HistoricalDefectCase
|   |   +-- generate_cases.py      # Generates 180 synthetic cases (30 per class)
|   |   +-- ingest_cases.py        # Embeds cases -> ChromaDB pcb_defect_cases
|   |   +-- ingest_standards.py    # Embeds 18 IPC excerpts -> pcb_standards
|   |   +-- test_retrieval.py      # ChromaDB sanity-check queries
|   |
|   +-- retrieval/                 # Phase 3: RAG Retrieval Pipeline
|   |   +-- schemas.py             # RetrievalQuery, RetrievalResult, AnalysisResult
|   |   +-- query_builder.py       # Detection JSON -> typed RetrievalQuery objects
|   |   +-- retriever.py           # PCBRetriever: ChromaDB queries + combining
|   |   +-- confidence.py          # Confidence scoring + human-review flagging
|   |
|   +-- generation/                # Phase 4: LLM Report Generation
|   |   +-- schemas.py             # InspectionReport, DefectReport, ... (output contract)
|   |   +-- prompts.py             # System/user prompts, bbox->location, fallback templates
|   |   +-- generator.py           # ReportGenerator: Ollama call + Pydantic validation + fallback
|   |
|   +-- api/                       # REST API
|   |   +-- main.py                # FastAPI app: /health, /detect, /analyze, /report, /report/health
|   |   +-- schemas.py             # API request/response Pydantic models
|   |
|   +-- gui/                       # Streamlit front-end
|       +-- app.py                 # Upload/webcam -> /analyze -> annotated results
|
+-- .claude/
|   +-- launch.json                # Dev server configs ("api", "gui")
|
+-- models/
|   +-- best.pt                          # Active checkpoint the API loads = v3 fine-tune (gitignored)
|   +-- best_deeppcb_only.pt             # Backup: pre-fine-tune, DeepPCB-only checkpoint
|   +-- best_finetune_v1_undertrained.pt # Backup: 15-epoch fine-tune checkpoint
|   +-- best_finetune_v2.pt              # Backup: 35-epoch fine-tune checkpoint
|   +-- training_results.json            # Metrics for whichever run produced current best.pt
|   +-- training_results_deeppcb_only.json
|
+-- chroma_db/
|   +-- chroma.sqlite3             # Persistent vector store (gitignored)
|
+-- scripts/                       # Audit + demo utilities (not production code)
|   +-- audit_run.py               # Full-system audit: data, training, ChromaDB
|   +-- audit_api_detect.py        # API edge-case testing
|   +-- demo_query_builder.py      # Prints QueryPlan for sample detections
|   +-- verify_grounding.py        # Phase 4: checks report is grounded in retrieved cases
|
+-- tests/
|   +-- test_retrieval.py          # 30+ unit + integration tests for Phase 3
|   +-- test_generation.py         # 27 unit + 1 integration test for Phase 4 (mocked LLM)
|
+-- notebooks/                     # Exploratory notebooks (not production code)
```

---

## 🔍 Defect Classes

Six bare-board fabrication defects from DeepPCB, grounded in IPC-A-610 terminology:

| ID | Class | Description | Common Root Cause |
|----|-------|-------------|------------------|
| 0 | `open` | Broken copper trace — conductor discontinuity | Under-etch or mechanical scoring during fabrication |
| 1 | `short` | Unintended copper bridge between separate nets | Over-etch boundary failure or copper contamination |
| 2 | `mousebite` | Localised edge notch from lateral under-etch | Etch chemistry imbalance at conductor edges |
| 3 | `spur` | Unwanted copper protrusion into clearance zone | Resist adhesion failure during pattern plating |
| 4 | `copper` | Spurious copper island in mask-free area | Plating bath contamination or resist pinholes |
| 5 | `pin-hole` | Through-void in copper plane or via barrel | Hydrogen entrapment during electroplating |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Git
- NVIDIA GPU with CUDA (optional — CPU works, training will be significantly slower)

### 1. Clone and Install

```bash
git clone <repository-url>
cd PCB_faultDetection

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> All commands must be run from the project root (`PCB_faultDetection/`) so `src` package imports resolve correctly.

### 2. DeepPCB Dataset Setup

Clone [DeepPCB](https://github.com/tangsanli5201/DeepPCB) directly into place:

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB
```

This gives you `data/raw/DeepPCB/PCBData/`. See [DEEPPCB_ASSUMPTIONS.md](DEEPPCB_ASSUMPTIONS.md) for the on-disk layout — `prepare_data.py` handles the real DeepPCB structure automatically (annotations in sibling `{id}_not/` folders, space-separated label lines).

```bash
# Convert DeepPCB annotations to YOLO format with stratified 80/10/10 split
python -m src.vision.prepare_data

# With overrides
python -m src.vision.prepare_data \
  --raw-dir data/raw/DeepPCB/PCBData \
  --output-dir data/processed \
  
  --seed 42 \
  --force
```

Expected output: `data/processed/` with `images/{train,val,test}/`, matching `labels/`, and `data.yaml`. Typical counts: ~1,200 train / ~150 val / ~150 test.

### 3. Train the Model

```bash
# Fast — lightweight backbone, good for development
python -m src.vision.train --model yolov8n.pt --epochs 50 --batch 16

# Higher accuracy
python -m src.vision.train --model yolov8s.pt --epochs 50 --batch 16

# Limited GPU VRAM (e.g. 4 GB) — lower the batch to avoid CUDA out-of-memory
python -m src.vision.train --model yolov8s.pt --epochs 50 --batch 8
```

> **GPU vs CPU:** training is dramatically faster on an NVIDIA GPU (~2–3 h for 50 epochs) than on CPU (~90 h). Install the CUDA build of PyTorch first — see the GPU note in [Quick Start](#-quick-start-end-to-end) step 1. Ultralytics auto-selects the GPU and prints `CUDA:0 (<your GPU>)` at startup. If VRAM is tight, drop `--batch` (and optionally `--imgsz 512`).

Output:
- `models/best.pt` — best checkpoint by mAP50
- `models/training_results.json` — per-class mAP, precision, recall
- `models/deeppcb/` — Ultralytics artifacts (confusion matrix, loss curves)

Expected mAP@0.5: 85–95% depending on backbone. If any class is below 70% recall, check that class's annotation count in the prepared data.

### 4. Build the Case Database

```bash
# Generate 180 synthetic historical defect cases (30 per class)
python -m src.data.generate_cases
# -> data/cases/historical_cases.json

# Embed and store in ChromaDB
python -m src.data.ingest_cases
# -> Collection: pcb_defect_cases (180 vectors)

# Embed IPC standards excerpts
python -m src.data.ingest_standards
# -> Collection: pcb_standards (18 vectors)
```

### 5. Verify

```bash
# Sanity check: queries ChromaDB and prints top matches
python -m src.data.test_retrieval

# Custom query
python -m src.data.test_retrieval --query "open trace from under-etch"

# Full test suite (requires populated ChromaDB)
pytest tests/test_retrieval.py -v

# Unit tests only — no ChromaDB needed
pytest tests/test_retrieval.py -v -m "not integration"
```

---

## 📸 Real-Photo Fine-Tuning (VOC_PCB)

### Why this exists

The DeepPCB-only model (above) is trained exclusively on synthetic, grayscale, template-difference renders. It has **zero color/lighting sensitivity issue** — it was verified to still detect defects correctly on a color-tinted DeepPCB image — but it also has **zero exposure to real camera photos**: no sensor noise, no realistic lighting, no perspective. A genuine photo of a bare board (let alone an assembled, soldermask-coated board) produces no useful detections, because the visual domain is completely unlike anything in DeepPCB.

`VOC_PCB.zip` (the PKU "PCB-DATASET" in Pascal VOC format) closes that gap: it's 10,668 **real RGB photographs** of bare boards shot under 12 lighting setups, annotated with the same 6 defect concepts as DeepPCB under different names.

### The dataset has two problems, both handled by `prepare_voc_pcb.py`

1. **75% redundant.** 8,001 of the 10,668 images are pure flip/rotation copies of the other 2,667 "base" images — same board, same lighting, zero new information, and redundant with Ultralytics' own live flip/rotation augmentation. Only the 2,667 base images (covering 521 unique physical crops across ~5 lighting variants each) are extracted and used — cutting disk usage from ~1.2 GB to ~350 MB with **no loss of real signal**.
2. **Leaky official split.** The dataset's own `train.txt`/`test.txt` mixes augmented copies of the same physical crop across different splits (verified: 515 overlapping base identities). `prepare_voc_pcb.py` instead splits at the **crop-group level** — every lighting variant of one physical crop stays in a single split — so validation numbers are trustworthy.

### Running it

```bash
# 1. Place VOC_PCB.zip at the project root (or pass --zip)
python -m src.vision.prepare_voc_pcb
# -> extracts 2,667 base images to data/raw/VOC_PCB/
# -> writes data/processed_finetune/ (leak-free split + a 400-image DeepPCB "replay buffer"
#    mixed into train, to protect against forgetting the synthetic domain)

# 2. Fine-tune from the existing DeepPCB checkpoint (not from scratch)
python -m src.vision.train \
  --data data/processed_finetune/data.yaml \
  --model models/best.pt \
  --epochs 20 \
  --batch 4 \
  --freeze 10 \
  --lr0 0.001 \
  --workers 4 \
  --name pcb_finetune_real
```

- `--freeze 10` freezes the first 10 backbone layers (standard YOLOv8 transfer-learning practice) — cheaper, and protects already-learned low-level features.
- `--lr0 0.001` is a reduced learning rate appropriate for continuing training on a pretrained checkpoint rather than training fresh. Lower it further (e.g. `0.0005`) when continuing from an already-fine-tuned checkpoint for additional epochs.
- `--workers 4` (default is 8) works around an intermittent Windows `FileNotFoundError` from DataLoader worker processes under heavy parallel disk I/O — lower it further if it recurs.
- This mutates `models/best.pt` like any other training run. **Back up first** if you want to keep the current checkpoint: `cp models/best.pt models/best_<label>.pt`.
- Watch `mAP50` in the console output across epochs — if it's still climbing at the final epoch (not plateauing), rerun with more epochs starting from the resulting checkpoint rather than treating it as done.

### Results (v3 checkpoint — 50 total epochs: 15 initial + 20 + 15 continuation, `lr0` lowered to 0.0005 for the final 15)

The fine-tune was done in three passes, each continuing from the previous and validated on held-out test sets. The current `models/best.pt` is the v3 checkpoint; earlier passes are kept as `models/best_finetune_v1_undertrained.pt` and `models/best_finetune_v2.pt`.

**Real-photo capability gained** — held-out VOC_PCB test set (crops never seen during training):

| Metric | v2 (35 ep) | **v3 (50 ep, live)** |
|---|---|---|
| mAP@0.5 | 0.711 | **0.788** |
| mAP@0.5–0.95 | 0.348 | **0.395** |
| Precision | 0.830 | **0.897** |
| Recall | 0.632 | **0.691** |

Per-class AP50 (v3):

| Class | AP50 |
|---|---|
| pin-hole | 0.989 |
| short | 0.873 |
| open | 0.822 |
| mousebite | 0.702 |
| spur | 0.689 |
| copper | 0.655 |

**Synthetic-domain retention** — same DeepPCB test set used to validate the original model, checking whether the replay buffer prevented catastrophic forgetting:

| Metric | Before fine-tune | v3 (live) |
|---|---|---|
| mAP@0.5 | 0.985 | 0.973 |
| mAP@0.5–0.95 | 0.584 | 0.547 |
| Precision | 0.982 | 0.962 |
| Recall | 0.973 | 0.940 |

Retention is nearly complete — only ~1 point of mAP50 lost vs. the original DeepPCB-only model, while gaining real-photo capability that didn't exist before. The replay buffer worked as intended.

**Weakest classes** (copper 0.655, spur 0.689, mousebite 0.702 on real photos) are the natural next target if you want to push further: more real photos of those specific defect types, or additional fine-tuning epochs, would likely help most. Note all three improved markedly from v2 (copper 0.567→0.655, mousebite 0.574→0.702), so continued training is still paying off.

---

## 📝 Report Generation (Phase 4 · Local LLM)

Phase 4 turns a Phase 3 `AnalysisResult` into a structured, cited `InspectionReport` — the final intelligence layer before the frontend. It uses a **local** LLM via [Ollama](https://ollama.com) (no API key, offline, no per-call cost), and is **fallback-safe**: if the LLM is unavailable it produces schema-identical reports from static templates.

### Setup (optional — the system runs without it)

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull the model (run in any terminal — it is a global CLI, no venv needed):
ollama pull llama3.2
# 3. Restart the API. It auto-detects the model at startup.
```

Default model is **`llama3.2`** (3B, ~2 GB — fits a 4 GB GPU and runs fast). Override with the `OLLAMA_MODEL` env var (e.g. `qwen2.5:3b`, `llama3.1:8b`) or `OLLAMA_HOST` for a remote Ollama. Without Ollama, `/report` still returns valid reports using fallback templates.

### How it decides LLM vs. fallback

Per defect, in `ReportGenerator.generate_report()`:

| Condition | Mode |
|---|---|
| Ollama reachable **and** model pulled **and** `retrieval_confidence != "uncertain"` | **LLM** (`generated_by="llm"`) |
| LLM unavailable, call fails, output fails Pydantic validation, or band is `uncertain` | **fallback** (`generated_by="fallback"`, `root_cause.unsupported=true`) |

`temperature=0.2` (structured output, not creative). Every LLM response is JSON-parsed **and** Pydantic-validated before use — raw text never reaches the client. `generate_report()` never raises.

### Grounding — the primary quality constraint

A report that invents root causes not in the retrieved context is worse than no report. Grounding is enforced three ways:

1. **Prompt** — the model is told to use *only* the retrieved cases/standards, to cite only listed `case_id`s in `evidence_basis`, and to set `unsupported=true` when no case clears the `0.55` similarity threshold (imported from `confidence.py`, not hardcoded).
2. **Validation** — malformed or off-schema output is rejected and falls back.
3. **Verification script** — check any generated report against its retrieval context:

```bash
python scripts/verify_grounding.py --image path/to/pcb.jpg
```

It flags invented `case_id`s, primary causes sharing no vocabulary with any retrieved case, and `unsupported=false` on empty retrieval, then prints a grounding score. Run it before trusting LLM output on a new prompt or model. (In fallback mode it reports "grounded by construction" rather than a hollow 100%.)

### Report shape (`InspectionReport`)

```
report_id, generated_at, total_defects, requires_human_review, overall_severity,
defect_reports[]:
    defect_class, location ("top-left" ... "centre"), 
    severity {level, score 1-5, rationale, ipc_reference?},
    root_cause {primary_cause, contributing_factors[≤3], confidence, evidence_basis[case_ids], unsupported},
    corrective_action {immediate, process_adjustment, re_inspection, ipc_reference?},
    narrative, generated_by
generation_metadata {model_used?, generation_mode, prompt_tokens_used?, total_defects_processed, fallback_count}
```

See the [API Reference](#-api-reference) for the `/report` and `/report/health` endpoints.

---

## ▶️ Running the API

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs`.

> First request after startup is slow (~25–30s) — `PCBRetriever` eagerly loads the sentence-transformer model and ChromaDB client at process start, not on first use. Subsequent requests are fast.

---

## 🖥️ Running the GUI

A Streamlit front-end (`src/gui/app.py`) for single-shot analysis: upload a PCB image or take one webcam snapshot, then view detections, retrieved historical cases, standards excerpts, and confidence scoring — all in the browser. It calls the same `/analyze` endpoint documented below, so the API must be running first.

```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — GUI
streamlit run src/gui/app.py
```

Open `http://localhost:8501`. If you're using `.claude/launch.json`-based tooling, the `api` and `gui` configurations are pre-defined.

**What it shows:**
- Sidebar: API URL override, live `/health` check (model-loaded status), confidence-band color legend
- Image source: file upload (JPEG/PNG) or webcam snapshot (single frame, not continuous video)
- Annotated image with bounding boxes colored by `retrieval_confidence` band (green=high, yellow=medium, orange=low, red=uncertain)
- Per-defect cards: detection confidence, top case similarity, cases/standards found, human-review warning, and expandable historical case + standards excerpt tabs

**Notes:**
- The GUI is a thin client — all detection/retrieval/scoring logic lives in the API, so `/analyze`'s behavior (503s when the model or ChromaDB isn't ready, zero-detection handling, etc.) surfaces identically in the UI.
- Requires `models/best.pt` to exist (trained model) and `chroma_db/` to be populated — without them the sidebar reports the same "not loaded" / 503 states as the raw API.

---

## 📡 API Reference

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

If `model_loaded` is `false`, `/detect` and `/analyze` return HTTP 503. The model auto-reloads when `models/best.pt` changes on disk — no server restart needed after retraining.

`/health` does not currently report ChromaDB collection state — `/analyze` returns 503 separately if `chroma_db/` or the `pcb_defect_cases` collection is missing.

---

### `POST /detect`

Upload a PCB image, receive detected defects with bounding boxes.

**Request:** `multipart/form-data`, field `file`. Accepted: JPEG, PNG. Max: 10 MB.

```bash
curl -X POST http://127.0.0.1:8000/detect -F "file=@path/to/pcb_image.jpg"
```

```json
{
  "detections": [
    {
      "defect_class": "open",
      "confidence": 0.82,
      "bbox": [100.5, 200.3, 150.2, 250.1]
    }
  ],
  "image_width": 640,
  "image_height": 640
}
```

`bbox` is `[x1, y1, x2, y2]` in original image pixel space. Zero detections returns `{"detections": [], ...}` — not an error.

| Status | Cause |
|--------|-------|
| 400 | Invalid file type, failed magic-byte check, or oversized file |
| 503 | Model weights not loaded |

---

### `POST /analyze`

Full pipeline: detection + RAG retrieval + confidence scoring.

```bash
curl -X POST http://127.0.0.1:8000/analyze -F "file=@path/to/pcb_image.jpg"
```

```json
{
  "total_detections": 1,
  "image_width": 640,
  "image_height": 640,
  "results": [
    {
      "detection": {
        "defect_class": "open",
        "confidence": 0.82,
        "confidence_band": "high",
        "bbox": [100.5, 200.3, 150.2, 250.1],
        "image_width": 640,
        "image_height": 640
      },
      "retrieved_cases": [
        {
          "case_id": "PCB-CASE-0042",
          "defect_class": "open",
          "component_type": "signal trace (outer layer)",
          "root_cause": "Under-etch during subtractive etching process...",
          "corrective_action": "Increased etch time monitoring with inline AOI...",
          "severity": 4,
          "outcome_notes": "Defect rate reduced by 60% after process adjustment",
          "similarity_score": 0.87
        }
      ],
      "retrieved_standards": [
        {
          "section_id": "IPC-A610-S3-01",
          "source_doc": "IPC-A-610 Rev H, Section 6.2.1",
          "defect_class": "open",
          "excerpt": "Open circuit — conductor continuity requirement...",
          "severity_threshold": 4,
          "relevance_score": 0.91
        }
      ],
      "retrieval_metadata": {
        "cases_found": 5,
        "standards_found": 3,
        "retrieval_confidence": "high",
        "flagged_for_human_review": false,
        "standards_skipped": false,
        "top_case_similarity": 0.87,
        "uncertain_detection": false
      }
    }
  ]
}
```

**`retrieval_confidence` levels:**

| Level | Condition |
|-------|-----------|
| `high` | Top case similarity > 0.80 AND detection confidence > 0.60 |
| `medium` | Top case similarity > 0.65 OR detection confidence > 0.60 (not both) |
| `low` | Top case similarity < 0.65 AND detection confidence < 0.60 |
| `uncertain` | Detection confidence < 0.40 |

**`flagged_for_human_review: true` when any of:**
- Detection confidence < 0.40
- Fewer than 2 historical cases retrieved
- Top case similarity < 0.55

---

### `GET /report/health`

Whether the LLM report generator is live — lets the frontend show a fallback-mode indicator.

```bash
curl http://127.0.0.1:8000/report/health
```

```json
{ "llm_available": true, "model": "llama3.2", "fallback_mode": false }
```

When Ollama isn't running or the model isn't pulled: `{"llm_available": false, "model": null, "fallback_mode": true}`.

---

### `POST /report`

Full pipeline **plus** LLM report generation: detection → retrieval → confidence → structured inspection report. Same request/validation as `/detect` and `/analyze` (`multipart/form-data`, field `file`, JPEG/PNG, ≤10 MB; optional `conf` and `iou` query params).

```bash
curl -X POST http://127.0.0.1:8000/report -F "file=@path/to/pcb_image.jpg"
```

```json
{
  "report_id": "1f0c…",
  "generated_at": "2026-07-09T01:46:48",
  "total_defects": 1,
  "requires_human_review": false,
  "overall_severity": "major",
  "defect_reports": [
    {
      "defect_class": "open",
      "location": "top-left",
      "severity": {"level": "major", "score": 4, "rationale": "…", "ipc_reference": "IPC-A-610 6.2.1"},
      "root_cause": {
        "primary_cause": "Under-etch severed the trace at the panel edge.",
        "contributing_factors": ["photoresist lift-off"],
        "confidence": "high",
        "evidence_basis": ["PCB-CASE-0009"],
        "unsupported": false
      },
      "corrective_action": {"immediate": "Quarantine board.", "process_adjustment": "Tighten etch control.", "re_inspection": "AOI + continuity.", "ipc_reference": null},
      "narrative": "An open was detected at the top-left of the board…",
      "generated_by": "llm"
    }
  ],
  "generation_metadata": {"model_used": "llama3.2", "generation_mode": "llm", "prompt_tokens_used": 812, "total_defects_processed": 1, "fallback_count": 0}
}
```

`generated_by` / `generation_mode` are `"fallback"` (and `root_cause.unsupported: true`) when the LLM is unavailable or a defect's band is `uncertain` — the JSON shape is identical either way. `/report` always returns a valid report; it never 500s on LLM errors.

| Status | Cause |
|--------|-------|
| 400 | Invalid or corrupted image |
| 415 | Unsupported file type / failed magic-byte check |
| 413 | File exceeds 10 MB |
| 503 | Model weights not loaded, or (with detections present) ChromaDB unavailable |

---

## ⚙️ Configuration Reference

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `LOW_CONFIDENCE_THRESHOLD` | `src/retrieval/query_builder.py` | `0.40` | Below this, detection is flagged uncertain |
| `HIGH_SIMILARITY_THRESHOLD` | `src/retrieval/confidence.py` | `0.80` | Above this, case match is high confidence |
| `MEDIUM_SIMILARITY_THRESHOLD` | `src/retrieval/confidence.py` | `0.65` | Above this, case match is medium confidence |
| `MIN_ACCEPTABLE_SIMILARITY` | `src/retrieval/confidence.py` | `0.55` | Below this, flag for human review |
| `MIN_CASES_FOR_CONFIDENT_REPORT` | `src/retrieval/confidence.py` | `2` | Fewer results triggers review flag |
| `default_cases_top_k` | `PCBRetriever.__init__`, `src/retrieval/retriever.py` | `5` | Historical cases returned per defect (constructor param, not a module constant) |
| `default_standards_top_k` | `PCBRetriever.__init__`, `src/retrieval/retriever.py` | `3` | Standards excerpts returned per defect (constructor param, not a module constant) |
| `EMBEDDING_MODEL` | `src/data/ingest_cases.py` | `all-MiniLM-L6-v2` | Must match across ingestion and retrieval |
| `COLLECTION_NAME` | `src/data/ingest_cases.py` | `pcb_defect_cases` | ChromaDB collection for cases (imported as `CASES_COLLECTION` in `retriever.py`) |
| `STANDARDS_COLLECTION` | `src/retrieval/retriever.py` | `pcb_standards` | ChromaDB collection for standards |
| `DEFAULT_BEST_WEIGHTS` | `src/vision/constants.py` | `models/best.pt` | YOLO checkpoint path |
| `MAX_UPLOAD_BYTES` | `src/api/main.py` | `10 * 1024 * 1024` | API file size limit, stored in bytes |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | `src/vision/prepare_data.py` | `0.8` / `0.1` / `0.1` | Stratified split ratios |
| `--seed` (CLI flag) | `src/vision/prepare_data.py` `main()` | `42` | Seed for reproducible train/val/test split — no module-level constant, only the CLI default |
| `VOC_PCB_CLASS_NAME_MAP` | `src/vision/constants.py` | see source | Maps VOC_PCB's class names (`open_circuit`, `mouse_bite`, `spurious_copper`, `missing_hole`, ...) to the DeepPCB taxonomy above |
| `DEFAULT_VOC_ZIP` / `DEFAULT_VOC_RAW_DIR` / `DEFAULT_FINETUNE_PROCESSED_DIR` | `src/vision/constants.py` | `VOC_PCB.zip` / `data/raw/VOC_PCB` / `data/processed_finetune` | Paths for the real-photo fine-tuning pipeline |
| `DEFAULT_REPLAY_COUNT` | `src/vision/prepare_voc_pcb.py` | `400` | DeepPCB train images mixed into the fine-tuning train split to guard against forgetting |
| `--freeze` / `--lr0` / `--workers` (CLI flags) | `src/vision/train.py` `main()` | `None` (Ultralytics defaults apply) | Optional fine-tuning overrides: freeze first N backbone layers, override initial learning rate, override DataLoader worker count |
| `DEFAULT_MODEL` (constructor `model`) | `src/generation/generator.py` | `llama3.2` | Ollama model for report generation; overridable via `OLLAMA_MODEL` env var |
| `OLLAMA_MODEL` / `OLLAMA_HOST` (env vars) | environment | unset → `llama3.2` / `localhost:11434` | Override the LLM model / point at a remote Ollama server |
| `temperature` / `max_tokens` (constructor) | `src/generation/generator.py` | `0.2` / `1500` | Low temperature is intentional — report generation is structured, not creative |
| Fallback similarity gate | `src/generation/prompts.py` | imported `HUMAN_REVIEW_SIMILARITY_THRESHOLD` (`0.55`) | Below this the LLM must mark `root_cause.unsupported=true` — imported from `confidence.py`, not redefined |

---

## 🗺 Phase Roadmap

- [x] **Phase 1 — Vision Foundation** — YOLOv8 fine-tuned on DeepPCB, `/detect` live, bounding boxes in original pixel space
- [x] **Phase 2 — Historical Case Database** — 180 synthetic cases + 18 IPC standards excerpts embedded in ChromaDB
- [x] **Phase 3 — RAG Retrieval Pipeline** — `/analyze` endpoint, per-class metadata filtering, confidence scoring, human-review flagging
- [x] **Phase 3.5 — GUI** — Streamlit app (`src/gui/app.py`) for upload/webcam single-shot analysis against `/analyze`
- [x] **Phase 1.5 — Real-Photo Fine-Tuning** — VOC_PCB real-camera dataset fine-tuned on top of the DeepPCB checkpoint; mAP50 0.79 on real bare-board photos (v3, 50 epochs) with only ~1 point of DeepPCB regression
- [x] **Phase 4 — Report Generation** — `/report` endpoint; local-LLM (Ollama) structured inspection report from `AnalysisResult`, Pydantic-validated, grounded in retrieved cases, fallback-safe
- [ ] **Phase 5 — Frontend & Dashboard** — Live/continuous video analysis, defect trend analytics, richer dashboard
- [ ] **Phase 6 — Validation & Feedback Loop** — Inspector feedback capture, retrieval re-ranking from confirmed root causes

---

## 🛠 Development Notes

### Known Limitations

- **Synthetic case database** — 180 cases are generated, not real inspection records. Root causes are grounded in real PCB manufacturing failure modes but not validated by domain experts.
- **Bare-board fabrication scope only** — even after VOC_PCB fine-tuning, the model only recognizes the 6 bare-copper fabrication defects. It was verified directly against a real photo of an assembled, soldermask-coated board and produced zero detections — assembled-board/AOI defects (solder bridges, tombstoning, missing components, misalignment) are a fundamentally different problem (different visual domain, different defect taxonomy) and out of scope without a new dataset and retraining effort.
- **Uneven real-photo performance** — after fine-tuning (v3), `copper`/`spur`/`mousebite` sit at 0.66–0.70 AP50 on real photos vs. 0.82–0.99 for `open`/`short`/`pin-hole`. See [Real-Photo Fine-Tuning](#-real-photo-fine-tuning-voc_pcb) for the full breakdown and how to extend it further.
- **Standards excerpts** — 18 documents modelled on IPC-A-610 / IPC-6012 structure; not verbatim from official publications.
- **Class imbalance** — `pin-hole` is the rarest class in DeepPCB. Low recall on this class after training is expected; check `training_results.json` per-class metrics.
- **LLM report quality depends on the local model** — a small local model (default `llama3.2` 3B) produces adequate grounded reports, but not frontier-model prose. Reports only reach `generated_by="llm"` when Ollama is running and the model is pulled; otherwise they are static fallbacks. Always run `verify_grounding.py` after changing the model or prompt.

### Adding a New Defect Class

1. Add class name + ID to `src/vision/constants.py`
2. Re-run `prepare_data.py` with updated annotation data
3. Retrain — new class included automatically
4. Add 30 synthetic cases in `generate_cases.py` and re-run `ingest_cases.py`
5. Add 3 IPC standard excerpts in `ingest_standards.py` and re-run
6. Add a fallback entry for the class to `FALLBACK_TEMPLATES` in `src/generation/prompts.py` (an assertion there fails fast if a class is missing)
7. Update the defect classes table in this README

### Swapping the Embedding Model

The embedding model is set in `src/data/ingest_cases.py` and must be identical in `src/retrieval/retriever.py`. If you change it after ingestion, delete `chroma_db/` and re-run both ingest scripts — mixing models in the same ChromaDB collection produces silently wrong similarity scores.

### Adding a New Standards Document

Add your excerpt to `src/data/ingest_standards.py`, then re-run:

```bash
python -m src.data.ingest_standards
```

---

## 🧪 Testing

Tests need the dev dependencies (not required to run the app): `pip install -r requirements-dev.txt`.

```bash
# All tests (Phase 3 + Phase 4). Phase 3 integration needs ChromaDB; Phase 4
# integration needs a running Ollama — both auto-skip when unavailable.
pytest tests/ -v

# Unit tests only — no ChromaDB, no LLM, always runnable offline
pytest tests/ -v -m "not integration"

# Phase 4 report-generation tests specifically (27 unit tests, mocked LLM)
pytest tests/test_generation.py -v -m "not integration"

# Grounding verification (Phase 4) — run before trusting LLM output on a new prompt/model
python scripts/verify_grounding.py --image path/to/pcb.jpg

# API edge-case audit / full-system audit / QueryPlan demo
python scripts/audit_api_detect.py
python scripts/audit_run.py
python scripts/demo_query_builder.py

# Per-class confidence threshold sweep (find optimal thresholds empirically)
python scripts/sweep_confidence.py --data data/processed_finetune/data.yaml --split test
```

Phase 3 coverage: single/multi-defect retrieval, zero-detection response, low-confidence flagging, no-match handling, standards retrieval. Phase 4 coverage: bbox→location (all 9 grid cells + edge cases), fallback for all 6 classes, LLM valid/invalid-JSON/exception paths, `requires_human_review` and `overall_severity` aggregation, and a grounding integration test (evidence_basis ⊆ retrieved case_ids).

---

## 🤝 Contributing

**Branch naming:** `phase-N/description` for phase work, `fix/description` for bug fixes.

**PR checklist:**
- [ ] Tests pass: `pytest tests/ -v -m "not integration"`
- [ ] No hardcoded paths — use constants from `src/vision/constants.py`
- [ ] Type hints on all new functions
- [ ] If retrieval logic changed: verify retrieval logs for 3 sample queries
- [ ] If embedding model changed: delete `chroma_db/` and re-ingest before testing

---

## 📚 Citation

If you use this project in research, please cite the DeepPCB dataset:

```bibtex
@misc{tang2019deeppcb,
  title         = {Deep PCB: A Dataset for PCB Defect Detection},
  author        = {Shaoyuan Tang and Fan He and Xiaolin Huang and Jie Yang},
  year          = {2019},
  eprint        = {1902.06197},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

IPC-A-610 and IPC-6012 terminology used in SOP documents are property of IPC — Association Connecting Electronics Industries.

The real-photo fine-tuning set is derived from `VOC_PCB.zip`, a Pascal VOC–format PCB defect dataset (commonly distributed as "PCB-DATASET" / PKU-Market-PCB). Verify the license terms of your specific copy before any commercial or redistributed use — no formal citation is included here since provenance wasn't independently verified for this project.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>PCB Defect Report Generator</strong><br>
  <em>YOLOv8 · RAG · ChromaDB · Ollama · FastAPI · DeepPCB</em>
</p>