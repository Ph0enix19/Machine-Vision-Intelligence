from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import BEAN_CLASSES, find_mvi_task1_files
from dashboard.adapters.task1_sources import import_task1_module
from dashboard.results_schema import make_task_result
from dashboard.vision_tasks import bean_class_from_colour, contour_measurements, contours_from_mask, draw_label, foreground_mask


class HemdanTask1ExternalAdapter(BaseInspectionAdapter):
    name = "Hemdan — I. Seed Classification — Classical OpenCV Task 1"
    member = "Hemdan"
    task_id = "I"
    task_name = "Seed Classification"
    method_name = "Classical OpenCV Task 1"
    task_type = "Classical Image Processing"
    description = "Uses the detected MVI_Task1 classical OpenCV classification approach without modifying the original file."
    main_outputs = ("class name", "class count", "contour area", "bounding box")

    def __init__(self) -> None:
        self.task1_files = find_mvi_task1_files()
        self.source_file = self._choose_source(self.task1_files)
        self._source_module = None
        self._source_load_error = ""

    def is_available(self) -> bool:
        return self.source_file is not None

    def availability_message(self) -> str:
        if self.source_file is None:
            return "No Python file found in MVI_Task1 or ../MVI_Task1."
        if self._source_load_error:
            return f"Using dashboard fallback because Hemdan Task 1 import failed: {self._source_load_error}"
        if len(self.task1_files) > 1:
            return f"Available. Multiple Task 1 files found; using {self.source_file.name}."
        return f"Available. Detected {self.source_file}"

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        source_result = self._process_with_source_file(image_bgr)
        if source_result is not None:
            return source_result
        return self._process_with_dashboard_fallback(image_bgr, **options)

    def _process_with_source_file(self, image_bgr: np.ndarray) -> dict[str, Any] | None:
        source = self._load_source_module()
        if source is None or not hasattr(source, "classify_frame") or not hasattr(source, "draw_results"):
            return None

        start = time.perf_counter()
        beans, _green_mask, _raw_mask, _final_mask = source.classify_frame(image_bgr.copy())
        annotated = source.draw_results(image_bgr.copy(), beans)
        detections: list[dict[str, Any]] = []
        for bean in beans:
            detections.append(
                {
                    "id": len(detections) + 1,
                    "class_name": bean.get("class"),
                    "confidence": None,
                    "area": float(bean.get("area", 0)),
                    "box_x": int(bean.get("x", 0)),
                    "box_y": int(bean.get("y", 0)),
                    "box_width": int(bean.get("w", 0)),
                    "box_height": int(bean.get("h", 0)),
                    "mask_area": float(bean.get("area", 0)),
                }
            )

        counts = Counter(detection["class_name"] for detection in detections)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_detected": len(detections),
                "white_kidney_count": counts.get("White Kidney Bean", 0),
                "speckled_kidney_count": counts.get("Speckled Kidney Bean", 0),
                "dark_kidney_count": counts.get("Dark Kidney Bean", 0),
                "average_confidence": None,
                "primary_result": f"{len([label for label in BEAN_CLASSES if counts.get(label, 0)])} bean class(es) detected",
            },
            task_outputs={
                "class_counts": {label: int(counts.get(label, 0)) for label in BEAN_CLASSES},
                "source_file": str(self.source_file) if self.source_file else "",
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "source_file": str(self.source_file) if self.source_file else "",
                "source_mode": "Imported Hemdan_Task1.py classify_frame/draw_results",
            },
        )

    def _process_with_dashboard_fallback(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        start = time.perf_counter()
        min_area = int(options.get("min_area", max(4000, image_bgr.shape[0] * image_bgr.shape[1] * 0.0004)))
        mask = foreground_mask(image_bgr)
        contours = contours_from_mask(image_bgr, mask, min_area)
        annotated = image_bgr.copy()
        detections: list[dict[str, Any]] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            measurements = contour_measurements(contour)
            if float(measurements["area"]) < min_area:
                continue
            class_name = bean_class_from_colour(image_bgr, contour)
            colour = _class_colour(class_name)
            cv2.drawContours(annotated, [contour], -1, colour, 2)
            cv2.rectangle(
                annotated,
                (int(measurements["box_x"]), int(measurements["box_y"])),
                (int(measurements["box_x"] + measurements["box_width"]), int(measurements["box_y"] + measurements["box_height"])),
                colour,
                2,
            )
            draw_label(annotated, class_name, int(measurements["box_x"]), int(measurements["box_y"]), colour)
            detections.append(
                {
                    "id": len(detections) + 1,
                    "class_name": class_name,
                    "confidence": None,
                    "area": measurements["area"],
                    "box_x": measurements["box_x"],
                    "box_y": measurements["box_y"],
                    "box_width": measurements["box_width"],
                    "box_height": measurements["box_height"],
                    "mask_area": measurements["area"],
                }
            )

        counts = Counter(detection["class_name"] for detection in detections)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_detected": len(detections),
                "white_kidney_count": counts.get("White Kidney Bean", 0),
                "speckled_kidney_count": counts.get("Speckled Kidney Bean", 0),
                "dark_kidney_count": counts.get("Dark Kidney Bean", 0),
                "average_confidence": None,
                "primary_result": f"{len([label for label in BEAN_CLASSES if counts.get(label, 0)])} bean class(es) detected",
            },
            task_outputs={
                "class_counts": {label: int(counts.get(label, 0)) for label in BEAN_CLASSES},
                "source_file": str(self.source_file) if self.source_file else "",
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "source_file": str(self.source_file) if self.source_file else "",
                "source_mode": "Dashboard fallback",
            },
        )

    def _load_source_module(self):
        if self.source_file is None:
            return None
        if self._source_module is not None:
            return self._source_module
        try:
            self._source_module = import_task1_module(self.source_file, "mvi_hemdan_task1_source")
            return self._source_module
        except Exception as exc:
            self._source_load_error = str(exc)
            return None

    def _choose_source(self, files: list[Path]) -> Path | None:
        if not files:
            return None
        hemdan_files = [path for path in files if "hemdan" in path.name.lower() or "hemdan" in str(path.parent).lower()]
        if hemdan_files:
            return sorted(hemdan_files)[0]
        task1_files = [path for path in files if "task1" in path.name.lower()]
        return sorted(task1_files)[0] if task1_files else sorted(files)[0]


def _class_colour(class_name: str) -> tuple[int, int, int]:
    return {
        "White Kidney Bean": (240, 240, 240),
        "Speckled Kidney Bean": (0, 210, 255),
        "Dark Kidney Bean": (40, 40, 220),
    }.get(class_name, (0, 180, 255))
