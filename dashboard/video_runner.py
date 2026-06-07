from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import pandas as pd

from dashboard.config import VIDEO_PREVIEW_EVERY_N_FRAMES
from dashboard.results_schema import build_video_result, compact_rows, detections_to_rows, ensure_result


ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[[dict[str, Any], int], None]


def process_video(
    input_path: Path,
    adapter: Any,
    output_video_path: Path,
    output_csv_path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    preview_every: int = VIDEO_PREVIEW_EVERY_N_FRAMES,
    max_frames: int | None = None,
    preview_only: bool = False,
    **adapter_options: Any,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid frame dimensions.")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not preview_only:
        writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_video_path}")

    all_rows: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            result = ensure_result(adapter.process_frame(frame, **adapter_options), default_frame=frame)
            last_result = result
            annotated = result.get("annotated_frame") if result.get("annotated_frame") is not None else frame
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)
            if writer is not None:
                writer.write(annotated)

            for row in detections_to_rows(result, source=getattr(adapter, "name", "")):
                row["frame"] = frame_index
                row["time_seconds"] = frame_index / fps if fps else None
                all_rows.append(row)

            if progress_callback:
                progress_callback(frame_index, total_frames)
            if preview_callback and (frame_index == 1 or frame_index % max(preview_every, 1) == 0):
                preview_callback(result, frame_index)
            if max_frames and frame_index >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    compact = compact_rows(all_rows)
    pd.DataFrame(compact).to_csv(output_csv_path, index=False)
    final_result = None
    latest_rows: list[dict[str, Any]] = []
    if last_result is not None:
        latest_rows = compact_rows(detections_to_rows(last_result, source=getattr(adapter, "name", "")))
        for row in latest_rows:
            row["frame"] = frame_index
            row["time_seconds"] = frame_index / fps if fps else None
        final_result = build_video_result(
            template_result=last_result,
            rows=latest_rows,
            annotated_frame=last_result.get("annotated_frame"),
            frames_processed=frame_index,
            fps=fps,
            summary_mode="latest_frame",
        )
    return {
        "frames_processed": frame_index,
        "fps": fps,
        "output_video": output_video_path,
        "output_csv": output_csv_path,
        "rows": compact,
        "latest_rows": latest_rows,
        "last_result": last_result,
        "final_result": final_result,
        "preview_only": preview_only,
    }
