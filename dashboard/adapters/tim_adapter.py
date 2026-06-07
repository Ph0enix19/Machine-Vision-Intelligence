from __future__ import annotations

import importlib.util
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import DEFAULT_IMG_SIZE, TIM_YOLO_WEIGHTS
from dashboard.results_schema import make_task_result, make_unavailable_result
from dashboard.vision_tasks import contour_from_box, draw_label, texture_record


class TimAdapter(BaseInspectionAdapter):
    name = "Tim — V. Texture Inspection"
    member = "Tim"
    task_id = "V"
    task_name = "Texture Inspection"
    method_name = "YOLO Seed Detection + OpenCV Texture Analysis"
    task_type = "AI + Classical Image Processing"
    description = "Uses Tim's extracted YOLO model for seed localization, then computes texture statistics per seed."
    main_outputs = ("texture label", "texture score", "surface pattern", "irregularity")

    def __init__(self, weights_path: Path = TIM_YOLO_WEIGHTS) -> None:
        self.weights_path = weights_path
        self._model = None
        self._load_error = ""

    def is_available(self) -> bool:
        return self.weights_path.exists() and importlib.util.find_spec("ultralytics") is not None

    def availability_message(self) -> str:
        if not self.weights_path.exists():
            return f"Unavailable: missing Tim weights at {self.weights_path}"
        if importlib.util.find_spec("ultralytics") is None:
            return "Unavailable: missing dependency ultralytics."
        if self._load_error:
            return self._load_error
        return "Available"

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path))
        return self._model

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        start = time.perf_counter()
        confidence = float(options.get("confidence", 0.60))
        img_size = int(options.get("img_size", DEFAULT_IMG_SIZE))
        device = str(options.get("device", "")).strip()
        try:
            model = self._load_model()
            predict_args: dict[str, Any] = {"source": image_bgr, "conf": confidence, "imgsz": img_size, "verbose": False}
            if device:
                predict_args["device"] = device
            result = model.predict(**predict_args)[0]
        except Exception as exc:
            self._load_error = str(exc)
            return make_unavailable_result(
                annotated_frame=image_bgr.copy(),
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason=str(exc),
            )

        annotated = image_bgr.copy()
        detections = _texture_detections_from_yolo(image_bgr, annotated, result)
        texture_counts = Counter(row["texture_label"] for row in detections)
        scores = [float(row["texture_score"]) for row in detections if row.get("texture_score") is not None]
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_analyzed": len(detections),
                "smooth_count": texture_counts.get("Smooth", 0),
                "medium_texture_count": texture_counts.get("Medium", 0),
                "rough_count": texture_counts.get("Rough", 0),
                "irregular_texture_count": texture_counts.get("Irregular", 0),
                "average_texture_score": mean(scores) if scores else None,
            },
            task_outputs={"texture_counts": dict(texture_counts)},
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "confidence": confidence,
                "weights": str(self.weights_path),
            },
        )


def _texture_detections_from_yolo(image_bgr: np.ndarray, annotated: np.ndarray, result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy().tolist() if boxes.conf is not None else [None] * len(xyxy)
    masks_xy = getattr(getattr(result, "masks", None), "xy", None)
    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = [int(value) for value in box]
        if masks_xy is not None and index - 1 < len(masks_xy) and len(masks_xy[index - 1]) >= 3:
            contour = np.asarray(masks_xy[index - 1], dtype=np.int32).reshape(-1, 1, 2)
        else:
            contour = contour_from_box(x1, y1, x2, y2)
        record = texture_record(image_bgr, contour, float(confidences[index - 1]) if confidences[index - 1] is not None else None)
        record["id"] = index
        detections.append(record)
        colour = (0, 180, 0) if record["texture_label"] == "Smooth" else (0, 165, 255) if record["texture_label"] == "Medium" else (0, 0, 220)
        cv2.drawContours(annotated, [contour], -1, colour, 2)
        draw_label(annotated, f"{record['texture_label']} {record['texture_score']:.2f}", x1, y1, colour)
    return detections
