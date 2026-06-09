from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from dashboard.adapters.ali_adapter import AliAdapter
from dashboard.results_schema import detections_to_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT.parent / "output_videos" / "ali test_classified.mp4"


def load_test_frame(path: Path):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        return frame

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read the first video frame: {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ali's dashboard YOLO adapter on one local frame.")
    parser.add_argument("media", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    media_path = args.media.resolve()
    print(f"Input: {media_path}")
    frame = load_test_frame(media_path)
    print(f"Frame shape: {frame.shape}")

    adapter = AliAdapter()
    print(f"Adapter: {adapter.name}")
    print(f"Available: {adapter.is_available()}")
    print(f"Availability message: {adapter.availability_message()}")
    print(f"Weights: {adapter.weights_path}")

    result = adapter.process_image(
        frame,
        confidence=args.confidence,
        img_size=args.img_size,
        device=args.device,
    )
    print(f"Result keys: {sorted(result.keys())}")
    print("Summary:")
    print(json.dumps(result.get("summary", {}), indent=2))
    print("Task outputs:")
    print(json.dumps(result.get("task_outputs", {}), indent=2))
    print("First detection:")
    first = (result.get("detections") or [None])[0]
    print(json.dumps(first, indent=2))

    error = result.get("metadata", {}).get("error")
    if error:
        print(f"ERROR: {error}")
        return 1

    labels = {
        detection.get("class_name")
        for detection in result.get("detections", [])
        if detection.get("class_name")
    }
    unexpected = labels.difference({"MATURE", "IMMATURE"})
    if unexpected:
        print(f"ERROR: unexpected Ali class tags: {sorted(unexpected)}")
        return 1
    if result.get("task_outputs", {}).get("original_model_tags") != ["IMMATURE", "MATURE"]:
        print(f"ERROR: original model tags changed: {result.get('task_outputs', {}).get('original_model_tags')}")
        return 1
    rows = detections_to_rows(result)
    if any(row.get("original_tag") != row.get("class_name") for row in rows):
        print("ERROR: Streamlit rows did not preserve Ali's original tag.")
        return 1
    print(f"Ali class tags returned: {sorted(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
