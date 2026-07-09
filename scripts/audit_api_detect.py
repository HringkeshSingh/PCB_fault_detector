"""Audit API and detect.py edge cases without requiring a trained model where possible."""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app, get_detector
from src.vision.constants import DEFAULT_BEST_WEIGHTS


def main() -> None:
    client = TestClient(app)

    print("=== API EDGE CASES ===")
    r = client.get("/health")
    print(f"GET /health: {r.status_code} {r.json()}")

    r = client.post("/detect")
    print(f"POST /detect no file: {r.status_code} (expect 422)")

    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    print(f"POST /detect text file: {r.status_code} {r.json().get('detail','')[:60]}")

    r = client.post("/detect", files={"file": ("fake.jpg", b"not an image", "image/jpeg")})
    print(f"POST /detect fake jpg bytes: {r.status_code} {r.json().get('detail','')[:60]}")

    big = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    r = client.post("/detect", files={"file": ("big.jpg", big, "image/jpeg")})
    print(f"POST /detect 11MB: {r.status_code} {r.json().get('detail','')[:60]}")

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (128, 128, 128)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/detect", files={"file": ("tiny.jpg", buf.read(), "image/jpeg")})
    print(f"POST /detect 1x1 image: {r.status_code} body={r.json() if r.status_code==200 else r.json().get('detail','')[:80]}")

    if DEFAULT_BEST_WEIGHTS.is_file():
        from src.vision.detect import DefectDetector
        det = DefectDetector()
        d1 = get_detector()
        d2 = get_detector()
        print(f"\nModel singleton same object: {d1 is d2}")

        blank = Image.new("RGB", (640, 640), (255, 255, 255))
        print(f"Blank 640: {det.detect(blank).model_dump()}")

        # Non-square resize test
        wide = Image.new("RGB", (1280, 640), (200, 200, 200))
        res = det.detect(wide)
        print(f"Wide 1280x640 dims: {res.image_width}x{res.image_height} dets={len(res.detections)}")
        for d in res.detections[:3]:
            print(f"  bbox={d.bbox} (max x should be <=1280)")
    else:
        print(f"\nSKIP inference tests — no weights at {DEFAULT_BEST_WEIGHTS}")


if __name__ == "__main__":
    main()
