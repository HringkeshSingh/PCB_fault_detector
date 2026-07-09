"""Quick unit checks for YOLO conversion (no dataset required)."""

from src.vision.prepare_data import deeppcb_line_to_yolo


def test_deeppcb_to_yolo_center() -> None:
    # Box 100,100 -> 200,200 on 640x640, class 2 (short -> YOLO idx 1)
    line = "100,100,200,200,2"
    yolo = deeppcb_line_to_yolo(line)
    assert yolo is not None
    parts = yolo.split()
    assert parts[0] == "1"
    assert abs(float(parts[1]) - 0.234375) < 1e-4  # center x = 150/640
    assert abs(float(parts[2]) - 0.234375) < 1e-4
    assert abs(float(parts[3]) - 0.15625) < 1e-4   # width 100/640
    assert abs(float(parts[4]) - 0.15625) < 1e-4


def test_invalid_line_returns_none() -> None:
    assert deeppcb_line_to_yolo("bad line") is None
    assert deeppcb_line_to_yolo("1,2,3,4,99") is None


if __name__ == "__main__":
    test_deeppcb_to_yolo_center()
    test_invalid_line_returns_none()
    print("prepare_data conversion checks passed.")
