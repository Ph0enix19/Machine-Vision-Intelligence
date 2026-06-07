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
from dashboard.results_schema import make_task_result, make_unavailable_result


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
            return f"Unavailable: Hemdan Task 1 import failed: {self._source_load_error}"
        if len(self.task1_files) > 1:
            return f"Available. Multiple Task 1 files found; using {self.source_file.name}."
        return f"Available. Detected {self.source_file}"

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        source_result = self._process_with_source_file(image_bgr)
        if source_result is not None:
            return source_result
        return make_unavailable_result(
            annotated_frame=image_bgr.copy(),
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            reason=self._source_load_error or "Hemdan Task 1 source does not expose classify_frame and draw_results.",
        )

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
