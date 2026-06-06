from __future__ import annotations

import base64
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
import requests

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import HANY_ROBOFLOW_API_KEY, HANY_ROBOFLOW_MODEL_ID
from dashboard.results_schema import make_task_result
from dashboard.vision_tasks import (
    colour_health_record,
    contour_from_box,
    contour_measurements,
    contours_from_mask,
    draw_label,
    foreground_mask,
)


class HanyAdapter(BaseInspectionAdapter):
    name = "Hany — IV. Maturity & Health Condition"
    member = "Hany"
    task_id = "IV"
    task_name = "Maturity and Health Condition"
    method_name = "Roboflow Detection + HSV/RGB Color Analysis"
    task_type = "AI + Classical Image Processing"
    description = "Uses Hany's Roboflow model when configured, then applies local RGB/HSV maturity and health analysis with an OpenCV fallback."
    main_outputs = ("RGB", "HSV", "color uniformity", "discoloration", "maturity", "health")

    def __init__(self) -> None:
        self._inference_error = ""

    def availability_message(self) -> str:
        if not HANY_ROBOFLOW_API_KEY:
            return "Available with OpenCV fallback. Add MVI_HANY_ROBOFLOW_API_KEY to enable Roboflow."
        if self._inference_error:
            return f"Available with OpenCV fallback. Roboflow warning: {self._inference_error}"
        return f"Available with Roboflow model {HANY_ROBOFLOW_MODEL_ID}."

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        start = time.perf_counter()
        confidence = float(options.get("confidence", 0.40))
        min_area = int(options.get("min_area", max(250, image_bgr.shape[0] * image_bgr.shape[1] * 0.0004)))
        annotated = image_bgr.copy()
        predictions = self._infer(image_bgr, confidence)
        detections = self._detections_from_predictions(image_bgr, annotated, predictions, min_area)
        inference_mode = "Roboflow" if detections else "OpenCV fallback"

        if not detections:
            mask = foreground_mask(image_bgr)
            contours = contours_from_mask(image_bgr, mask, min_area)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                measurements = contour_measurements(contour)
                if float(measurements["area"]) < min_area:
                    continue
                record = colour_health_record(image_bgr, contour)
                record["id"] = len(detections) + 1
                record["roboflow_class"] = None
                record["confidence"] = None
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
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "inference_mode": inference_mode,
                "roboflow_model": HANY_ROBOFLOW_MODEL_ID if inference_mode == "Roboflow" else "",
                "roboflow_error": self._inference_error,
                "original_code": "group_members/hany/original/V3.py",
            },
        )

    def _infer(self, image_bgr: np.ndarray, confidence: float) -> list[dict[str, Any]]:
        if not HANY_ROBOFLOW_API_KEY:
            return []
        try:
            encoded_ok, encoded_image = cv2.imencode(".jpg", image_bgr)
            if not encoded_ok:
                self._inference_error = "Could not encode image for Roboflow."
                return []
            response = requests.post(
                f"https://detect.roboflow.com/{HANY_ROBOFLOW_MODEL_ID}",
                params={
                    "api_key": HANY_ROBOFLOW_API_KEY,
                    "confidence": int(confidence * 100),
                    "format": "json",
                },
                data=base64.b64encode(encoded_image).decode("ascii"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if not response.ok:
                self._inference_error = f"Roboflow returned HTTP {response.status_code}."
                return []
            result = response.json()
            self._inference_error = ""
            return [
                prediction
                for prediction in result.get("predictions", [])
                if float(prediction.get("confidence", 0.0)) >= confidence
            ]
        except requests.RequestException as exc:
            self._inference_error = f"Roboflow request failed: {type(exc).__name__}."
            return []
        except (TypeError, ValueError):
            self._inference_error = "Roboflow returned an invalid response."
            return []

    def _detections_from_predictions(
        self,
        image_bgr: np.ndarray,
        annotated: np.ndarray,
        predictions: list[dict[str, Any]],
        min_area: int,
    ) -> list[dict[str, Any]]:
        height, width = image_bgr.shape[:2]
        detections: list[dict[str, Any]] = []
        for prediction in predictions:
            box_width = max(1, int(prediction.get("width", 0)))
            box_height = max(1, int(prediction.get("height", 0)))
            center_x = int(prediction.get("x", 0))
            center_y = int(prediction.get("y", 0))
            x1 = max(0, center_x - box_width // 2)
            y1 = max(0, center_y - box_height // 2)
            x2 = min(width - 1, center_x + box_width // 2)
            y2 = min(height - 1, center_y + box_height // 2)
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < min_area:
                continue

            contour = contour_from_box(x1, y1, x2, y2)
            record = colour_health_record(image_bgr, contour)
            class_name = str(prediction.get("class", "object"))
            maturity_override, health_override = _labels_from_class(class_name)
            if maturity_override:
                record["maturity_label"] = maturity_override
            if health_override:
                record["health_label"] = health_override
                record["discoloration_status"] = "None" if health_override == "Healthy" else health_override
            record.update(
                {
                    "id": len(detections) + 1,
                    "roboflow_class": class_name,
                    "confidence": float(prediction.get("confidence", 0.0)),
                }
            )
            detections.append(record)

            colour = (0, 180, 0) if record["health_label"] == "Healthy" else (0, 0, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            draw_label(
                annotated,
                f"{class_name} {record['confidence']:.0%}",
                x1,
                y1,
                colour,
            )
        return detections


def _labels_from_class(class_name: str) -> tuple[str | None, str | None]:
    label = class_name.lower().replace("_", " ").replace("-", " ")
    maturity = None
    health = None
    if "semi" in label and ("mature" in label or "ripe" in label):
        maturity = "Semi-Mature"
    elif "immature" in label or "unripe" in label:
        maturity = "Immature"
    elif "overripe" in label:
        maturity = "Overripe"
    elif "mature" in label or "ripe" in label:
        maturity = "Mature"

    if any(term in label for term in ("defect", "damage", "disease", "unhealthy", "discolor", "mold", "crack", "broken")):
        health = "Defective"
    elif "healthy" in label:
        health = "Healthy"
    return maturity, health
