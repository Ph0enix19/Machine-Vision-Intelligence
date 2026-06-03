from __future__ import annotations

import time
from collections import Counter
from statistics import mean
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.adapters.task1_sources import find_member_task1_file, import_task1_module
from dashboard.results_schema import make_task_result, make_unavailable_result


class AliTask1Adapter(BaseInspectionAdapter):
    name = "Ali — III. Seed Growth Measurement — Classical OpenCV Task 1"
    member = "Ali"
    task_id = "III"
    task_name = "Seed Growth Inspection using Measurement"
    method_name = "Classical OpenCV Task 1 Measurement"
    task_type = "Classical Image Processing"
    description = "Uses Ali_Task1.py measurement and maturity rules safely without opening its file dialog."
    main_outputs = ("length", "width", "area", "perimeter", "aspect ratio", "shape")

    def __init__(self) -> None:
        self.source_file = find_member_task1_file("Ali")
        self._module = None
        self._load_error = ""

    def is_available(self) -> bool:
        return self.source_file is not None

    def availability_message(self) -> str:
        if self.source_file is None:
            return "Unavailable: Ali_Task1.py was not found in MVI_Task1."
        if self._load_error:
            return f"Available with fallback warning: {self._load_error}"
        return f"Available. Detected {self.source_file.name}."

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        module = self._load_module()
        if module is None:
            return make_unavailable_result(
                annotated_frame=image_bgr.copy(),
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason=self._load_error or "Ali Task 1 module could not be loaded.",
            )

        start = time.perf_counter()
        mask = module.preprocess(image_bgr.copy())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = image_bgr.copy()
        detections: list[dict[str, Any]] = []
        maturity_counts: Counter[str] = Counter()

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            features = module.extract_features(contour)
            if features is None:
                continue
            label, detail = _normalise_classification(module.classify(features))
            maturity_counts[label.title()] += 1
            module.draw_annotation(annotated, features, label, detail)
            detections.append(
                {
                    "id": len(detections) + 1,
                    "length": features.get("length"),
                    "width": features.get("width"),
                    "area": features.get("area"),
                    "perimeter": features.get("perimeter"),
                    "aspect_ratio": features.get("aspect_ratio"),
                    "circularity": features.get("circularity"),
                    "compactness": features.get("compactness"),
                    "equivalent_diameter": _equivalent_diameter(features.get("area")),
                    "shape": features.get("shape"),
                }
            )

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
            task_outputs={
                "measurement_units": "pixels",
                "shape_counts": dict(shapes),
                "maturity_rule_counts": dict(maturity_counts),
                "source_file": str(self.source_file),
            },
            detections=detections,
            metadata={"adapter": self.name, "fps": 1.0 / elapsed, "source_file": str(self.source_file)},
        )

    def _load_module(self):
        if self.source_file is None:
            return None
        if self._module is not None:
            return self._module
        try:
            self._module = import_task1_module(self.source_file, "mvi_ali_task1_source")
            return self._module
        except Exception as exc:
            self._load_error = str(exc)
            return None


def _normalise_classification(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple):
        label = str(value[0])
        detail = str(value[1]) if len(value) > 1 else ""
    else:
        label = str(value)
        detail = "rule"
    return label, detail


def _equivalent_diameter(area: Any) -> float | None:
    if area is None:
        return None
    return float(np.sqrt(4.0 * float(area) / np.pi))
