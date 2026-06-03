from __future__ import annotations

import time
from collections import Counter
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.results_schema import make_task_result
from dashboard.vision_tasks import colour_health_record, contour_measurements, contours_from_mask, draw_label, foreground_mask


class HanyAdapter(BaseInspectionAdapter):
    name = "Hany — IV. Maturity & Health Condition"
    member = "Hany"
    task_id = "IV"
    task_name = "Maturity and Health Condition"
    method_name = "HSV/RGB Color Analysis"
    task_type = "Classical Image Processing"
    description = "Safe maturity and health wrapper using RGB/HSV colour statistics. Hany's original MVS/Roboflow script is preserved and not imported."
    main_outputs = ("RGB", "HSV", "color uniformity", "discoloration", "maturity", "health")

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        start = time.perf_counter()
        min_area = int(options.get("min_area", max(250, image_bgr.shape[0] * image_bgr.shape[1] * 0.0004)))
        mask = foreground_mask(image_bgr)
        contours = contours_from_mask(image_bgr, mask, min_area)
        annotated = image_bgr.copy()
        detections: list[dict[str, Any]] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            measurements = contour_measurements(contour)
            if float(measurements["area"]) < min_area:
                continue
            record = colour_health_record(image_bgr, contour)
            record["id"] = len(detections) + 1
            detections.append(record)
            colour = (0, 180, 0) if record["health_label"] == "Healthy" else (0, 0, 220)
            cv2.drawContours(annotated, [contour], -1, colour, 2)
            draw_label(annotated, f"{record['maturity_label']} | {record['health_label']}", int(measurements["box_x"]), int(measurements["box_y"]), colour)

        maturity_counts = Counter(row["maturity_label"] for row in detections)
        health_counts = Counter(row["health_label"] for row in detections)
        discoloration_count = sum(1 for row in detections if row["discoloration_status"] != "None")
        dark_patch_count = sum(1 for row in detections if float(row["dark_patch_ratio"]) > 0.08)
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
            task_outputs={"maturity_counts": dict(maturity_counts), "health_counts": dict(health_counts)},
            detections=detections,
            metadata={"adapter": self.name, "fps": 1.0 / elapsed, "original_code": "group_members/hany/original/V3.py"},
        )

