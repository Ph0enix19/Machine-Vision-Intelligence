from __future__ import annotations

import platform
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from dashboard.results_schema import compact_rows, detections_to_rows, ensure_result


def bgr_to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray | None) -> np.ndarray | None:
    if image_rgb is None:
        return None
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def uploaded_image_to_bgr(uploaded_file: Any) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return rgb_to_bgr(np.array(image))


def detections_dataframe(result: dict[str, Any] | None, source: str = "") -> pd.DataFrame:
    rows = compact_rows(detections_to_rows(result, source=source))
    return pd.DataFrame(rows)


def dict_dataframe(values: dict[str, Any], key_name: str = "Label", value_name: str = "Value") -> pd.DataFrame:
    return pd.DataFrame([{key_name: key, value_name: value} for key, value in values.items()])


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    allowed = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "output"


def save_uploaded_temp(uploaded_file: Any, suffix: str) -> Path:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(uploaded_file.getbuffer())
        return Path(temp.name)
    finally:
        temp.close()


def open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system().lower() == "windows" else 0
    capture = cv2.VideoCapture(index, backend)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def check_webcam(index: int = 0) -> tuple[bool, str]:
    capture = open_camera(index)
    try:
        if not capture.isOpened():
            return False, f"Camera index {index} did not open."
        ok, _ = capture.read()
        if not ok:
            return False, f"Camera index {index} opened but did not return a frame."
        return True, f"Camera index {index} is available."
    finally:
        capture.release()


def call_adapter(adapter: Any, image_bgr: np.ndarray, *, frame: bool = False, **options: Any) -> dict[str, Any]:
    try:
        if frame:
            result = adapter.process_frame(image_bgr, **options)
        else:
            result = adapter.process_image(image_bgr, **options)
        return ensure_result(result, default_frame=image_bgr)
    except Exception as exc:
        annotated = image_bgr.copy()
        cv2.putText(
            annotated,
            f"Adapter error: {exc}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return {
            "annotated_frame": annotated,
            "member": getattr(adapter, "member", "Unknown"),
            "task_id": getattr(adapter, "task_id", ""),
            "task_name": getattr(adapter, "task_name", "Unknown Task"),
            "method": getattr(adapter, "method_name", "Unknown"),
            "summary": {"primary_result": "Adapter error"},
            "task_outputs": {"error": str(exc)},
            "detections": [],
            "metadata": {"error": str(exc)},
        }


def format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
