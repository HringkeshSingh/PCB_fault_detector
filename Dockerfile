# PCB Defect Report Generator — one image for both the FastAPI backend and the
# Streamlit GUI (docker-compose.yml picks the command). CPU-only PyTorch.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    ANONYMIZED_TELEMETRY=False

# OpenCV (pulled in by ultralytics) needs these shared libraries at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheel first so requirements.txt doesn't pull the much larger CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY data/cases/ data/cases/
COPY models/best.pt models/best.pt

# Build the ChromaDB store at image build time (also caches the embedding model),
# so containers start ready without a separate ingest step.
RUN python -m src.data.ingest_cases && python -m src.data.ingest_standards

RUN useradd --create-home --uid 1000 app && chown -R app:app /app /opt/hf-cache
USER app

# Embedding model is already cached above; don't call the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
