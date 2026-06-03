from __future__ import annotations

import time
from collections import Counter
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.adapters.task1_sources import find_member_task1_file
from dashboard.results_schema import make_task_result


COLOURS = {
    "Good": (0, 200, 0),
    "Mouldy": (0, 165, 255),
    "Broken": (0, 0, 220),
    "Cracked": (128, 0, 200),
    "Defective": (0, 220, 220),
}


class AdonaiTask1Adapter(BaseInspectionAdapter):
    name = "Adonai — II. Quality Inspection — Classical OpenCV Task 1"
    member = "Adonai"
    task_id = "II"
    task_name = "Quality Inspection"
    method_name = "Classical OpenCV Task 1 Quality Rules"
    task_type = "Classical Image Processing"
    description = "Safe wrapper copied from Adonai_Task1.py logic; the original file starts a video loop at import time."
    main_outputs = ("healthy/defective status", "crack", "broken", "moldy", "damaged")

    def __init__(self) -> None:
        self.source_file = find_member_task1_file("Adonai")

    def is_available(self) -> bool:
        return self.source_file is not None

    def availability_message(self) -> str:
        if self.source_file is None:
            return "Unavailable: Adonai_Task1.py was not found in MVI_Task1."
        return f"Available. Safely wrapped {self.source_file.name}."

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        start = time.perf_counter()
        frame_display = cv2.resize(image_bgr.copy(), (700, 540))
        gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        eq = cv2.equalizeHist(blur)
        binary = _segment(eq)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < 5000 or area > 50000:
                continue
            features, _ = _extract_features(frame_display, gray, contour)
            label = _classify(features)
            quality_status, defect_type = _quality_from_label(label)
            detections.append(
                {
                    "id": len(detections) + 1,
                    "quality_status": quality_status,
                    "defect_type": defect_type,
                    "crack": defect_type == "Crack",
                    "broken": defect_type == "Broken",
                    "moldy": defect_type == "Moldy",
                    "damaged": defect_type == "Damaged",
                    "area": features["area"],
                    "confidence": None,
                }
            )
            colour = COLOURS.get(label, (255, 255, 255))
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame_display, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        quality_counts = Counter(row["quality_status"] for row in detections)
        defect_counts = Counter(row["defect_type"] for row in detections)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=frame_display,
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
            task_outputs={
                "quality_counts": dict(quality_counts),
                "defect_type_counts": dict(defect_counts),
                "source_file": str(self.source_file),
            },
            detections=detections,
            metadata={"adapter": self.name, "fps": 1.0 / elapsed, "source_file": str(self.source_file)},
        )


def _segment(eq: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = cv2.bitwise_not(threshold)
    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)


def _extract_features(img: np.ndarray, gray: np.ndarray, contour: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    features = {
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "aspect_ratio": round(float(w) / h, 3) if h else 0,
        "extent": round(area / (w * h), 3) if w * h else 0,
        "solidity": round(area / hull_area, 3) if hull_area else 0,
        "circularity": round((4 * np.pi * area) / (perimeter**2), 3) if perimeter else 0,
    }
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    edges = cv2.Canny(gray, 30, 100)
    internal_edges = cv2.bitwise_and(edges, edges, mask=mask)
    bean_pixels = gray[mask == 255]
    features["crack_ratio"] = round(np.sum(internal_edges > 0) / area, 4) if area else 0
    features["mean_v"] = round(float(np.mean(bean_pixels)), 2) if len(bean_pixels) else 0
    features["std_v"] = round(float(np.std(bean_pixels)), 2) if len(bean_pixels) else 0
    dark = np.sum(bean_pixels < 80)
    features["dark_ratio"] = round(dark / len(bean_pixels), 4) if len(bean_pixels) else 0
    return features, mask


def _classify(features: dict[str, Any]) -> str:
    if features["std_v"] > 95 and features["dark_ratio"] < 0.62:
        return "Broken"
    if features["solidity"] < 0.970:
        return "Defective"
    if features["crack_ratio"] < 0.115:
        return "Good"
    if features["dark_ratio"] > 0.80:
        return "Cracked"
    return "Mouldy"


def _quality_from_label(label: str) -> tuple[str, str]:
    if label == "Good":
        return "Healthy", "None"
    mapping = {
        "Cracked": "Crack",
        "Broken": "Broken",
        "Mouldy": "Moldy",
        "Defective": "Damaged",
    }
    return "Defective", mapping.get(label, "Unknown")
