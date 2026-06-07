from __future__ import annotations

import importlib.util
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import ADONAI_YOLO_WEIGHTS, DEFAULT_IMG_SIZE
from dashboard.results_schema import make_task_result, make_unavailable_result
from dashboard.vision_tasks import contour_from_box, draw_label, quality_record


CLASS_CONF_THRESHOLD = {
    "good": 0.15,
    "cracked": 0.35,
    "broken": 0.40,
    "mouldy": 0.30,
    "defective": 0.40,
}

CLASS_BIAS = {
    "good": 3.8,
    "cracked": 1.5,
    "broken": 1.0,
    "mouldy": 1.0,
    "defective": 1.0,
}


class AdonaiAdapter(BaseInspectionAdapter):
    name = "Adonai — II. Quality Inspection"
    member = "Adonai"
    task_id = "II"
    task_name = "Quality Inspection"
    method_name = "CLAHE + YOLO Quality Detection"
    task_type = "AI / Neural Network"
    description = "Safe wrapper for Adonai's quality inspection concept. Original MVS/Tkinter script is preserved and not imported."
    main_outputs = ("healthy/defective status", "crack", "broken", "moldy", "damaged")

    def __init__(self, weights_path: Path = ADONAI_YOLO_WEIGHTS) -> None:
        self.weights_path = weights_path
        self._model = None
        self._load_error = ""
        self._clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))

    def is_available(self) -> bool:
        return self.weights_path.exists() and importlib.util.find_spec("ultralytics") is not None

    def availability_message(self) -> str:
        if not self.weights_path.exists():
            return f"Unavailable: missing Adonai quality weights at {self.weights_path}. Original script uses a machine-specific path."
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
        if not self.is_available():
            annotated = image_bgr.copy()
            reason = self.availability_message()
            cv2.putText(annotated, "Adonai quality model unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return make_unavailable_result(
                annotated_frame=annotated,
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason=reason,
            )

        start = time.perf_counter()
        confidence = float(options.get("confidence", 0.20))
        img_size = int(options.get("img_size", DEFAULT_IMG_SIZE))
        enhanced = self._apply_clahe(image_bgr)
        try:
            model = self._load_model()
            result = model.predict(source=enhanced, conf=confidence, iou=0.45, imgsz=img_size, verbose=False)[0]
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

        annotated = enhanced.copy()
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", model.names)
        if boxes is not None and boxes.cls is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
            confidences = boxes.conf.detach().cpu().numpy().tolist() if boxes.conf is not None else [None] * len(classes)
            for raw_index, class_id in enumerate(classes):
                class_name = _class_name(names, class_id)
                original_confidence = float(confidences[raw_index]) if confidences[raw_index] is not None else None
                adjusted_confidence = (
                    min(0.95, original_confidence * CLASS_BIAS.get(class_name.lower(), 1.0))
                    if original_confidence is not None
                    else None
                )
                if adjusted_confidence is not None and adjusted_confidence < CLASS_CONF_THRESHOLD.get(class_name.lower(), 0.40):
                    continue
                x1, y1, x2, y2 = [int(value) for value in xyxy[raw_index]]
                contour = contour_from_box(x1, y1, x2, y2)
                record = _quality_from_label(class_name, enhanced, contour, adjusted_confidence)
                record["id"] = len(detections) + 1
                detections.append(record)
                colour = (0, 180, 0) if record["quality_status"] == "Healthy" else (0, 0, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                draw_label(annotated, f"{class_name} {adjusted_confidence:.0%}", x1, y1, colour)

        quality_counts = Counter(row["quality_status"] for row in detections)
        defect_counts = Counter(row["defect_type"] for row in detections)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_inspected": len(detections),
                "healthy_count": quality_counts.get("Healthy", 0),
                "defective_count": quality_counts.get("Defective", 0),
                "crack_count": defect_counts.get("Crack", 0),
                "broken_count": defect_counts.get("Broken", 0),
                "moldy_count": defect_counts.get("Moldy", 0),
                "damaged_count": defect_counts.get("Damaged", 0),
                "unknown_count": defect_counts.get("Unknown", 0),
            },
            task_outputs={"quality_counts": dict(quality_counts), "defect_type_counts": dict(defect_counts)},
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "confidence": confidence,
                "weights": str(self.weights_path),
                "class_bias": CLASS_BIAS,
            },
        )

    def _apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_equalized = self._clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge([l_equalized, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _quality_from_label(class_name: str, image_bgr: np.ndarray, contour: np.ndarray, confidence: float | None) -> dict[str, Any]:
    lowered = class_name.lower()
    if "good" in lowered or "healthy" in lowered:
        record = quality_record(image_bgr, contour, confidence)
        record.update({"quality_status": "Healthy", "defect_type": "None", "crack": False, "broken": False, "moldy": False, "damaged": False})
        return record
    if "crack" in lowered:
        defect = "Crack"
    elif "broken" in lowered:
        defect = "Broken"
    elif "mould" in lowered or "mold" in lowered:
        defect = "Moldy"
    elif "damage" in lowered or "defect" in lowered:
        defect = "Damaged"
    else:
        defect = "Unknown"
    record = quality_record(image_bgr, contour, confidence)
    record.update(
        {
            "quality_status": "Defective" if defect != "Unknown" else "Unknown",
            "defect_type": defect,
            "crack": defect == "Crack",
            "broken": defect == "Broken",
            "moldy": defect == "Moldy",
            "damaged": defect == "Damaged",
        }
    )
    return record
