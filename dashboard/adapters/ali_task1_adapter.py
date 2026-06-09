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


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 1024


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
                reason=self._load_error or "Ali Task 1 module could not be loaded.",
            )

        start = time.perf_counter()
        processing_frame = _processing_frame(image_bgr)
        features_with_contours, segmentation_mode = _find_seed_features(module, processing_frame)
        annotated = image_bgr.copy()
        detections: list[dict[str, Any]] = []
        maturity_counts: Counter[str] = Counter()

        for contour, features in features_with_contours:
            label, detail = _normalise_classification(module.classify(features))
            maturity_counts[label] += 1
            _draw_annotation_on_input(annotated, contour, label, detail)
            detections.append(
                {
                    "id": len(detections) + 1,
                    "original_tag": label,
                    "class_name": label,
                    "classification_detail": detail,
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

        _draw_count_panel(annotated, maturity_counts)
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
                "mature_count": maturity_counts.get("MATURE", 0),
                "immature_count": maturity_counts.get("IMMATURE", 0),
                "detected_tags": ", ".join(sorted(maturity_counts)) or "None",
            },
            task_outputs={
                "measurement_units": f"reference pixels ({REFERENCE_WIDTH}x{REFERENCE_HEIGHT})",
                "shape_counts": dict(shapes),
                "class_counts": dict(maturity_counts),
                "segmentation_mode": segmentation_mode,
                "source_file": str(self.source_file),
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "source_file": str(self.source_file),
                "input_resolution": f"{image_bgr.shape[1]}x{image_bgr.shape[0]}",
                "processing_resolution": f"{REFERENCE_WIDTH}x{REFERENCE_HEIGHT}",
            },
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


def _processing_frame(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.shape[1] == REFERENCE_WIDTH and image_bgr.shape[0] == REFERENCE_HEIGHT:
        return image_bgr.copy()
    return cv2.resize(
        image_bgr,
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )


def _find_seed_features(module: Any, frame_bgr: np.ndarray) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], str]:
    original_mask = module.preprocess(frame_bgr.copy())
    original_features = _features_from_mask(module, original_mask)
    if original_features:
        return original_features, "Ali original preprocessing"

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fallback_candidates: list[tuple[list[tuple[np.ndarray, dict[str, Any]]], str]] = []
    for threshold_type, name in (
        (cv2.THRESH_BINARY_INV, "Otsu dark foreground fallback"),
        (cv2.THRESH_BINARY, "Otsu light foreground fallback"),
    ):
        _, mask = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        features = _features_from_mask(module, mask)
        if features:
            fallback_candidates.append((features, name))

    if not fallback_candidates:
        return [], "No valid seed contours"
    return max(fallback_candidates, key=lambda item: len(item[0]))


def _features_from_mask(module: Any, mask: np.ndarray) -> list[tuple[np.ndarray, dict[str, Any]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(mask.shape[0] * mask.shape[1])
    valid: list[tuple[np.ndarray, dict[str, Any]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        touches_most_edges = sum(
            (
                x <= 1,
                y <= 1,
                x + width >= mask.shape[1] - 1,
                y + height >= mask.shape[0] - 1,
            )
        ) >= 3
        if area >= frame_area * 0.85 or (touches_most_edges and area >= frame_area * 0.25):
            continue
        features = module.extract_features(contour)
        if features is not None:
            valid.append((contour, features))
    return valid


def _draw_annotation_on_input(
    annotated: np.ndarray,
    reference_contour: np.ndarray,
    label: str,
    detail: str,
) -> None:
    scale_x = annotated.shape[1] / REFERENCE_WIDTH
    scale_y = annotated.shape[0] / REFERENCE_HEIGHT
    contour = reference_contour.astype(np.float32).copy()
    contour[:, 0, 0] *= scale_x
    contour[:, 0, 1] *= scale_y
    contour = np.rint(contour).astype(np.int32)

    colour = (0, 200, 0) if label == "MATURE" else (0, 0, 220)
    cv2.drawContours(annotated, [contour], -1, colour, 2)
    x, y, _, _ = cv2.boundingRect(contour)
    text = f"{label} [{detail}]"
    font_scale = max(0.45, min(0.7, annotated.shape[1] / 1800.0))
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        1,
    )
    text_x = max(0, min(x, annotated.shape[1] - text_width - 4))
    text_y = max(text_height + baseline + 4, y - 8)
    cv2.rectangle(
        annotated,
        (text_x, text_y - text_height - baseline - 5),
        (text_x + text_width + 4, text_y + 2),
        colour,
        -1,
    )
    cv2.putText(
        annotated,
        text,
        (text_x + 2, text_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_count_panel(annotated: np.ndarray, counts: Counter[str]) -> None:
    panel_width = min(285, max(180, annotated.shape[1] - 16))
    panel_height = min(112, max(76, annotated.shape[0] - 16))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
    annotated[:] = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)
    font_scale = max(0.55, min(0.9, annotated.shape[1] / 1400.0))
    line_height = max(25, panel_height // 3)
    for index, (text, colour) in enumerate(
        (
            (f"MATURE: {counts.get('MATURE', 0)}", (0, 200, 0)),
            (f"IMMATURE: {counts.get('IMMATURE', 0)}", (0, 0, 220)),
            (f"TOTAL: {sum(counts.values())}", (255, 255, 255)),
        )
    ):
        cv2.putText(
            annotated,
            text,
            (16, 8 + line_height * (index + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            2,
            cv2.LINE_AA,
        )
