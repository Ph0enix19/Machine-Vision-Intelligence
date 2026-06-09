from __future__ import annotations

import base64
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
import requests

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.config import get_setting
from dashboard.results_schema import make_task_result, make_unavailable_result
from dashboard.vision_tasks import draw_label


class HanyAdapter(BaseInspectionAdapter):
    name = "Hany — IV. Maturity & Health Condition"
    member = "Hany"
    task_id = "IV"
    task_name = "Maturity and Health Condition"
    method_name = "Roboflow Hosted Detection"
    task_type = "AI / Hosted Inference"
    description = "Uses Hany's original Roboflow model behavior: hosted inference, class labels, confidence scores, and bounding boxes."
    main_outputs = ("Roboflow class", "confidence", "maturity", "health")

    def __init__(self) -> None:
        self._inference_error = ""
        self.api_key = get_setting("MVI_HANY_ROBOFLOW_API_KEY")
        self.model_id = get_setting("MVI_HANY_ROBOFLOW_MODEL_ID", "mvi-task-2-dqpn6/2")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def availability_message(self) -> str:
        if not self.api_key:
            return "Unavailable: add MVI_HANY_ROBOFLOW_API_KEY to enable Hany's Roboflow model."
        if self._inference_error:
            return f"Roboflow error: {self._inference_error}"
        return f"Available with Roboflow model {self.model_id}. Cloud secret detected."

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        if not self.api_key:
            return make_unavailable_result(
                annotated_frame=image_bgr.copy(),
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason="MVI_HANY_ROBOFLOW_API_KEY is not configured.",
            )

        start = time.perf_counter()
        confidence = float(options.get("confidence", 0.40))
        annotated = image_bgr.copy()
        predictions = self._infer(image_bgr, confidence)
        if self._inference_error:
            return make_unavailable_result(
                annotated_frame=annotated,
                member=self.member,
                task_id=self.task_id,
                task_name=self.task_name,
                method=self.method_name,
                reason=self._inference_error,
            )

        detections = self._detections_from_predictions(image_bgr, annotated, predictions)

        maturity_counts = Counter(row["maturity_label"] for row in detections if row.get("maturity_label"))
        health_counts = Counter(row["health_label"] for row in detections if row.get("health_label"))
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
                "discoloration_count": 0,
                "dark_patch_count": 0,
            },
            task_outputs={
                "maturity_counts": dict(maturity_counts),
                "health_counts": dict(health_counts),
                "roboflow_class_counts": dict(Counter(row["roboflow_class"] for row in detections)),
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "inference_mode": "Roboflow",
                "roboflow_model": self.model_id,
                "credential_status": "Configured",
                "original_code": "group_members/hany/original/V3.py",
            },
        )

    def _infer(self, image_bgr: np.ndarray, confidence: float) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        try:
            encoded_ok, encoded_image = cv2.imencode(".jpg", image_bgr)
            if not encoded_ok:
                self._inference_error = "Could not encode image for Roboflow."
                return []
            response = requests.post(
                f"https://detect.roboflow.com/{self.model_id}",
                params={
                    "api_key": self.api_key,
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
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = str(prediction.get("class", "object"))
            maturity_label, health_label = _labels_from_class(class_name)
            record = {
                "id": len(detections) + 1,
                "mean_r": None,
                "mean_g": None,
                "mean_b": None,
                "mean_h": None,
                "mean_s": None,
                "mean_v": None,
                "color_uniformity": None,
                "discoloration_status": None,
                "dark_patch_ratio": None,
                "maturity_label": maturity_label,
                "health_label": health_label,
                "roboflow_class": class_name,
                "confidence": float(prediction.get("confidence", 0.0)),
            }
            detections.append(record)

            colour = _class_colour(class_name)
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


def _class_colour(class_name: str) -> tuple[int, int, int]:
    palette = [
        (255, 82, 82),
        (82, 255, 121),
        (82, 182, 255),
        (255, 210, 82),
        (210, 82, 255),
    ]
    return palette[hash(class_name) % len(palette)]
