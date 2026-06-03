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
from dashboard.vision_tasks import contour_from_box, contour_measurements, contours_from_mask, draw_label, foreground_mask


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
        detections = _measurement_detections_from_yolo(annotated, result)
        fallback_used = False
        if not detections:
            detections = _measurement_detections_from_contours(image_bgr, annotated)
            fallback_used = True
        lengths = [float(row["length"]) for row in detections if row.get("length") is not None]
        widths = [float(row["width"]) for row in detections if row.get("width") is not None]
        areas = [float(row["area"]) for row in detections if row.get("area") is not None]
        ratios = [float(row["aspect_ratio"]) for row in detections if row.get("aspect_ratio") is not None]
        shapes = Counter(row["shape"] for row in detections if row.get("shape"))
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
            },
            task_outputs={"measurement_units": "pixels", "shape_counts": dict(shapes)},
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "confidence": confidence,
                "weights": str(self.weights_path),
                "original_code": "group_members/ali/original/live_seed_classifier.py",
                "fallback": "OpenCV contour measurement" if fallback_used else "",
            },
        )


def _measurement_detections_from_yolo(annotated: np.ndarray, result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    masks_xy = getattr(getattr(result, "masks", None), "xy", None)
    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = [int(value) for value in box]
        if masks_xy is not None and index - 1 < len(masks_xy) and len(masks_xy[index - 1]) >= 3:
            contour = np.asarray(masks_xy[index - 1], dtype=np.int32).reshape(-1, 1, 2)
        else:
            contour = contour_from_box(x1, y1, x2, y2)
        measurements = contour_measurements(contour)
        detections.append(
            {
                "id": index,
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
        colour = (20, 120, 220)
        cv2.drawContours(annotated, [contour], -1, colour, 2)
        draw_label(annotated, f"{measurements['shape']} L{measurements['length']:.0f} W{measurements['width']:.0f}", x1, y1, colour)
    return detections


def _measurement_detections_from_contours(image_bgr: np.ndarray, annotated: np.ndarray) -> list[dict[str, Any]]:
    min_area = max(250, image_bgr.shape[0] * image_bgr.shape[1] * 0.0004)
    mask = foreground_mask(image_bgr)
    contours = contours_from_mask(image_bgr, mask, int(min_area))
    detections: list[dict[str, Any]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        measurements = contour_measurements(contour)
        if float(measurements["area"]) < min_area:
            continue
        detections.append(
            {
                "id": len(detections) + 1,
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
        colour = (20, 120, 220)
        cv2.drawContours(annotated, [contour], -1, colour, 2)
        draw_label(
            annotated,
            f"{measurements['shape']} L{measurements['length']:.0f} W{measurements['width']:.0f}",
            int(measurements["box_x"]),
            int(measurements["box_y"]),
            colour,
        )
    return detections
