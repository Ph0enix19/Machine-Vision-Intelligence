from __future__ import annotations

import time
from collections import Counter
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.adapters.task1_sources import find_member_task1_file, import_task1_module
from dashboard.results_schema import make_task_result, make_unavailable_result


class HanyTask1Adapter(BaseInspectionAdapter):
    name = "Hany — IV. Maturity & Health Condition — Classical OpenCV Task 1"
    member = "Hany"
    task_id = "IV"
    task_name = "Maturity and Health Condition"
    method_name = "Classical OpenCV Task 1 Colour Analysis"
    task_type = "Classical Image Processing"
    description = "Uses Hany_Task1.py bean contour, maturity, and health analysis without starting its video loop."
    main_outputs = ("RGB", "HSV", "maturity", "health", "discoloration")

    def __init__(self) -> None:
        self.source_file = find_member_task1_file("Hany")
        self._module = None
        self._load_error = ""

    def is_available(self) -> bool:
        return self.source_file is not None

    def availability_message(self) -> str:
        if self.source_file is None:
            return "Unavailable: Hany_Task1.py was not found in MVI_Task1."
        if self._load_error:
            return f"Unavailable: {self._load_error}"
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
                reason=self._load_error or "Hany Task 1 module could not be loaded.",
            )

        start = time.perf_counter()
        contours, _mask = module.find_bean_contours(image_bgr.copy())
        contours = module.ContourSmoother().update(contours)
        beans = [module.analyse_bean(image_bgr.copy(), contour) for contour in contours]
        annotated = module.draw_results(image_bgr.copy(), beans, contours)
        detections: list[dict[str, Any]] = []

        for bean in beans:
            mean_r, mean_g, mean_b = bean.get("mean_rgb", (None, None, None))
            mean_h, mean_s, mean_v = bean.get("mean_hsv", (None, None, None))
            std_rgb = bean.get("std_rgb", ())
            dark_patch_ratio = _dark_patch_ratio(bean, image_bgr)
            detections.append(
                {
                    "id": len(detections) + 1,
                    "mean_r": mean_r,
                    "mean_g": mean_g,
                    "mean_b": mean_b,
                    "mean_h": mean_h,
                    "mean_s": mean_s,
                    "mean_v": mean_v,
                    "color_uniformity": float(sum(std_rgb) / len(std_rgb)) if std_rgb else None,
                    "discoloration_status": "None" if bean.get("quality") == "Healthy" else bean.get("quality"),
                    "dark_patch_ratio": dark_patch_ratio,
                    "maturity_label": _normalise_maturity(bean.get("maturity")),
                    "health_label": _normalise_health(bean.get("quality")),
                }
            )

        maturity_counts = Counter(row["maturity_label"] for row in detections if row.get("maturity_label"))
        health_counts = Counter(row["health_label"] for row in detections if row.get("health_label"))
        discoloration_count = sum(1 for row in detections if row.get("discoloration_status") not in {None, "", "None"})
        dark_patch_count = sum(1 for row in detections if row.get("dark_patch_ratio") is not None and float(row["dark_patch_ratio"]) > 0.08)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_analyzed": len(detections),
                "mature_count": maturity_counts.get("Mature", 0),
                "semi_mature_count": maturity_counts.get("Semi-Mature", 0),
                "immature_count": maturity_counts.get("Immature", 0),
                "overripe_count": maturity_counts.get("Overripe", 0),
                "healthy_count": health_counts.get("Healthy", 0),
                "discoloration_count": discoloration_count,
                "dark_patch_count": dark_patch_count,
            },
            task_outputs={
                "maturity_counts": dict(maturity_counts),
                "health_counts": dict(health_counts),
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
            self._module = import_task1_module(self.source_file, "mvi_hany_task1_source")
            return self._module
        except Exception as exc:
            self._load_error = str(exc)
            return None


def _normalise_maturity(label: Any) -> str:
    mapping = {"Ripe": "Mature", "Semi-Ripe": "Semi-Mature", "Unripe": "Immature"}
    return mapping.get(str(label), str(label))


def _normalise_health(label: Any) -> str:
    if str(label) == "Discolored":
        return "Discoloration"
    return str(label)


def _dark_patch_ratio(bean: dict[str, Any], image_bgr: np.ndarray) -> float | None:
    bbox = bean.get("bbox")
    if not bbox:
        return None
    x, y, w, h = bbox
    roi = image_bgr[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray < 65)) / float(gray.size)
