from __future__ import annotations

import platform
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

# Easy edit settings
MODEL_PATH = ROOT / "runs" / "segment" / "bean_seg_v1" / "weights" / "best.pt"
CAMERA_INDEX = 0
VIDEO_PATH = ""  # Optional: set to r"C:\path\to\test_video.mp4" to test from video
CONFIDENCE = 0.80
IMG_SIZE = 640

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
WINDOW_NAME = "Kidney Bean YOLO Segmentation"


def open_capture() -> cv2.VideoCapture | None:
    if VIDEO_PATH.strip():
        video_file = Path(VIDEO_PATH)
        if not video_file.exists():
            print(f"ERROR: Video file does not exist: {video_file}")
            return None
        print(f"Opening video file: {video_file}")
        return cv2.VideoCapture(str(video_file))

    backend = cv2.CAP_DSHOW if platform.system().lower() == "windows" else 0
    print(f"Opening webcam index {CAMERA_INDEX}")
    capture = cv2.VideoCapture(CAMERA_INDEX, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_counts(frame, counts: Counter[int], names, fps: float) -> None:
    rows = [f"FPS: {fps:.1f}"]
    if isinstance(names, dict):
        class_ids = sorted(int(key) for key in names.keys())
    else:
        class_ids = list(range(len(names)))

    for class_id in class_ids:
        rows.append(f"{class_name(names, class_id)}: {counts[class_id]}")

    x, y = 12, 12
    line_height = 28
    box_width = 360
    box_height = line_height * len(rows) + 16
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (20, 20, 20), -1)

    for index, text in enumerate(rows):
        text_y = y + 30 + index * line_height
        cv2.putText(
            frame,
            text,
            (x + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def counts_from_result(result) -> Counter[int]:
    counts: Counter[int] = Counter()
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return counts

    for class_id in boxes.cls.detach().cpu().numpy().astype(int).tolist():
        counts[class_id] += 1
    return counts


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: Missing model weights: {MODEL_PATH}")
        print("Train first with: python scripts/train_segmentation.py")
        return 1

    model = YOLO(str(MODEL_PATH))
    capture = open_capture()
    if capture is None or not capture.isOpened():
        print("ERROR: Could not open the webcam or video source.")
        if not VIDEO_PATH.strip():
            print("Try CAMERA_INDEX = 1 or 2, close apps using the camera, and check Windows privacy settings.")
        return 1

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"Capture opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    print("Press q to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    last_time = time.perf_counter()
    display_fps = 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            print("No more frames or failed to read from source.")
            break

        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False)
        result = results[0]
        annotated = result.plot()

        now = time.perf_counter()
        elapsed = max(now - last_time, 1e-6)
        last_time = now
        display_fps = (display_fps * 0.85) + ((1.0 / elapsed) * 0.15)

        counts = counts_from_result(result)
        draw_counts(annotated, counts, model.names, display_fps)

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
