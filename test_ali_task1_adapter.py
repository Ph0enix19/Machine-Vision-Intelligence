from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

from dashboard.adapters.ali_task1_adapter import AliTask1Adapter


ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEO = ROOT.parent / "output_videos" / "ali test_classified.mp4"


def first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read {path}")
    return frame


def main() -> int:
    tkinter_before = sys.modules.get("tkinter")
    adapter = AliTask1Adapter()
    assert adapter.is_available(), adapter.availability_message()
    assert "headless dashboard shim" in adapter.availability_message()
    assert sys.modules.get("tkinter") is tkinter_before
    source = first_frame(SAMPLE_VIDEO)
    expected = adapter.process_image(source)
    small = adapter.process_image(cv2.resize(source, (320, 240)))

    assert expected["detections"], "Full-resolution Ali sample produced no detections."
    assert len(small["detections"]) == len(expected["detections"]), (
        len(expected["detections"]),
        len(small["detections"]),
    )
    assert small["task_outputs"]["class_counts"] == expected["task_outputs"]["class_counts"]

    light_on_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(light_on_dark, (210, 240), (95, 55), -15, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(light_on_dark, (440, 235), (105, 60), 20, 0, 360, (220, 220, 220), -1)
    polarity_result = adapter.process_image(light_on_dark)
    assert len(polarity_result["detections"]) == 2, len(polarity_result["detections"])

    print("Ali Task 1 full-resolution detections:", len(expected["detections"]))
    print("Ali Task 1 320x240 detections:", len(small["detections"]))
    print("Resolution-normalized class counts:", small["task_outputs"]["class_counts"])
    print("Light-on-dark fallback detections:", len(polarity_result["detections"]))
    print("Light-on-dark segmentation mode:", polarity_result["task_outputs"]["segmentation_mode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
