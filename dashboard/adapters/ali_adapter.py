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
from dashboard.config import ALI_YOLO_WEIGHTS, DEFAULT_IMG_SIZE
from dashboard.results_schema import make_task_result, make_unavailable_result
from dashboard.vision_tasks import contour_from_box, contour_measurements


class AliAdapter(BaseInspectionAdapter):
    name = "Ali — III. Seed Growth Measurement"
    member = "Ali"
    task_id = "III"
    task_name = "Seed Growth Inspection using Measurement"
    method_name = "YOLO Segmentation + Pixel Measurement"
    task_type = "AI + Measurement"
    description = "Uses Ali's provided YOLO segmentation weights safely, then reports seed growth measurements in pixels."
    main_outputs = ("length", "width", "area", "perimeter", "aspect ratio", "circularity")

    def __init__(self, weights_path: Path = ALI_YOLO_WEIGHTS) -> None:
        self.weights_path = weights_path
        self._model = None
        self._load_error = ""

    def is_available(self) -> bool:
        return self.weights_path.exists() and importlib.util.find_spec("ultralytics") is not None

    def availability_message(self) -> str:
        if not self.weights_path.exists():
            return f"Unavailable: missing Ali weights at {self.weights_path}"
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
        confidence = float(options.get("confidence", 0.70))
        img_size = int(options.get("img_size", DEFAULT_IMG_SIZE))
        device = str(options.get("device", "")).strip()
        try:
            model = self._load_model()
            predict_args: dict[str, Any] = {
                "source": image_bgr,
                "conf": confidence,
                "iou": 0.30,
                "imgsz": img_size,
                "verbose": False,
            }
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
        names = getattr(result, "names", getattr(model, "names", {}))
        detections = _measurement_detections_from_yolo(annotated, result, names)
        lengths = [float(row["length"]) for row in detections if row.get("length") is not None]
        widths = [float(row["width"]) for row in detections if row.get("width") is not None]
        areas = [float(row["area"]) for row in detections if row.get("area") is not None]
        ratios = [float(row["aspect_ratio"]) for row in detections if row.get("aspect_ratio") is not None]
        shapes = Counter(row["shape"] for row in detections if row.get("shape"))
        class_counts = Counter(row["class_name"] for row in detections if row.get("class_name"))
        _draw_original_count_panel(annotated, class_counts)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_measured": len(detections),
                "average_length": mean(lengths) if lengths else None,
                "average_width": mean(widths) if widths else None,
                "average_area": mean(areas) if areas else None,
                "average_aspect_ratio": mean(ratios) if ratios else None,
                "most_common_shape": shapes.most_common(1)[0][0] if shapes else None,
                "mature_count": class_counts.get("MATURE", 0),
                "immature_count": class_counts.get("IMMATURE", 0),
                "detected_tags": ", ".join(sorted(class_counts)) or "None",
            },
            task_outputs={
                "measurement_units": "pixels",
                "original_model_tags": _ordered_model_tags(names),
                "class_counts": dict(class_counts),
                "shape_counts": dict(shapes),
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "confidence": confidence,
                "weights": str(self.weights_path),
                "original_code": "group_members/ali/original/live_seed_classifier.py",
            },
        )


def _measurement_detections_from_yolo(
    annotated: np.ndarray,
    result: Any,
    names: Any,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
    masks_xy = getattr(getattr(result, "masks", None), "xy", None)
    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = [int(value) for value in box]
        class_name = _class_name(names, int(class_ids[index - 1]))
        confidence = float(confidences[index - 1]) if confidences is not None else None
        if masks_xy is not None and index - 1 < len(masks_xy) and len(masks_xy[index - 1]) >= 3:
            contour = np.asarray(masks_xy[index - 1], dtype=np.int32).reshape(-1, 1, 2)
        else:
            contour = contour_from_box(x1, y1, x2, y2)
        measurements = contour_measurements(contour)
        detections.append(
            {
                "id": index,
                "original_tag": class_name,
                "class_name": class_name,
                "confidence": confidence,
                "length": measurements["length"],
                "width": measurements["width"],
                "area": measurements["area"],
                "perimeter": measurements["perimeter"],
                "aspect_ratio": measurements["aspect_ratio"],
                "circularity": measurements["circularity"],
                "compactness": measurements["compactness"],
                "equivalent_diameter": measurements["equivalent_diameter"],
                "shape": measurements["shape"],
            }
        )
        colour = _class_colour(class_name)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [contour], colour)
        annotated[:] = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
        cv2.polylines(annotated, [contour], isClosed=True, color=colour, thickness=3)
        confidence_text = f" {confidence:.2f}" if confidence is not None else ""
        _draw_original_tag(
            annotated,
            f"{class_name}{confidence_text}",
            x1,
            y1,
            colour,
        )
    return detections


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _class_colour(class_name: str) -> tuple[int, int, int]:
    if class_name == "MATURE":
        return (0, 255, 0)
    if class_name == "IMMATURE":
        return (0, 0, 255)
    return (255, 255, 255)


def _ordered_model_tags(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _draw_original_count_panel(annotated: np.ndarray, counts: Counter[str]) -> None:
    lines = (
        (f"MATURE: {counts.get('MATURE', 0)}", (0, 255, 0)),
        (f"IMMATURE: {counts.get('IMMATURE', 0)}", (0, 0, 255)),
        (f"TOTAL: {sum(counts.values())}", (255, 255, 255)),
    )
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (300, 132), (0, 0, 0), -1)
    annotated[:] = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
    for index, (text, colour) in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (20, 42 + index * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            colour,
            2,
            cv2.LINE_AA,
        )


def _draw_original_tag(
    annotated: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
) -> None:
    font_scale = max(0.55, min(0.8, annotated.shape[1] / 1600.0))
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2,
    )
    text_x = max(0, min(x, annotated.shape[1] - text_width - 4))
    text_y = max(text_height + baseline + 4, y - 10)
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        2,
        cv2.LINE_AA,
    )
