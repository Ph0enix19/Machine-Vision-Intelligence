from __future__ import annotations

import importlib.util
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import BEAN_CLASSES, DEFAULT_CONFIDENCE, DEFAULT_IMG_SIZE, HEMDAN_YOLO_WEIGHTS
from dashboard.results_schema import make_task_result, make_unavailable_result


class HemdanYoloAdapter(BaseInspectionAdapter):
    name = "Hemdan — I. Seed Classification — YOLO Segmentation"
    member = "Hemdan"
    task_id = "I"
    task_name = "Seed Classification"
    method_name = "YOLO Segmentation"
    task_type = "AI / Neural Network"
    description = "Local Ultralytics YOLO instance segmentation for kidney bean class identification."
    main_outputs = ("class name", "class count", "confidence", "mask/box area")

    def __init__(self, weights_path: Path = HEMDAN_YOLO_WEIGHTS) -> None:
        self.weights_path = weights_path
        self._model = None
        self._load_error = ""

    def is_available(self) -> bool:
        return self.weights_path.exists() and importlib.util.find_spec("ultralytics") is not None

    def availability_message(self) -> str:
        if not self.weights_path.exists():
            return f"Missing YOLO weights: {self.weights_path}"
        if importlib.util.find_spec("ultralytics") is None:
            return "Missing dependency: ultralytics"
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
        confidence = float(options.get("confidence", DEFAULT_CONFIDENCE))
        img_size = int(options.get("img_size", DEFAULT_IMG_SIZE))
        device = str(options.get("device", "")).strip()

        try:
            model = self._load_model()
            predict_args: dict[str, Any] = {
                "source": image_bgr,
                "imgsz": img_size,
                "conf": confidence,
                "verbose": False,
            }
            if device:
                predict_args["device"] = device
            result = model.predict(**predict_args)[0]
        except Exception as exc:
            self._load_error = str(exc)
            annotated = image_bgr.copy()
            cv2.putText(annotated, f"YOLO unavailable: {exc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return make_unavailable_result(
                annotated_frame=annotated,
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason=str(exc),
            )

        detections = self._detections_from_result(result, getattr(result, "names", getattr(model, "names", {})))
        counts = Counter(detection["class_name"] for detection in detections)
        confidences = [detection["confidence"] for detection in detections if detection.get("confidence") is not None]
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=result.plot(),
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_detected": len(detections),
                "white_kidney_count": counts.get("White Kidney Bean", 0),
                "speckled_kidney_count": counts.get("Speckled Kidney Bean", 0),
                "dark_kidney_count": counts.get("Dark Kidney Bean", 0),
                "average_confidence": float(sum(confidences) / len(confidences)) if confidences else None,
                "primary_result": f"{len([label for label in BEAN_CLASSES if counts.get(label, 0)])} bean class(es) detected",
            },
            task_outputs={"class_counts": {label: int(counts.get(label, 0)) for label in BEAN_CLASSES}},
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "confidence": confidence,
                "img_size": img_size,
                "weights": str(self.weights_path),
            },
        )

    def _detections_from_result(self, result: Any, names: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else []
        classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        confidences = boxes.conf.detach().cpu().numpy().tolist() if boxes.conf is not None else [None] * len(classes)
        masks_xy = getattr(getattr(result, "masks", None), "xy", None)
        detections: list[dict[str, Any]] = []

        for index, class_id in enumerate(classes, start=1):
            x1, y1, x2, y2 = [int(value) for value in xyxy[index - 1]]
            box_width = max(x2 - x1, 0)
            box_height = max(y2 - y1, 0)
            mask_area = None
            area = float(box_width * box_height)
            if masks_xy is not None and index - 1 < len(masks_xy) and len(masks_xy[index - 1]) >= 3:
                contour = np.asarray(masks_xy[index - 1], dtype=np.int32).reshape(-1, 1, 2)
                mask_area = float(cv2.contourArea(contour))
                area = mask_area
            detections.append(
                {
                    "id": index,
                    "class_name": _class_name(names, class_id),
                    "confidence": float(confidences[index - 1]) if confidences[index - 1] is not None else None,
                    "area": area,
                    "box_x": x1,
                    "box_y": y1,
                    "box_width": box_width,
                    "box_height": box_height,
                    "mask_area": mask_area,
                }
            )
        return detections


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)

