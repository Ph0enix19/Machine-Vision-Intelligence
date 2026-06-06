from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from dashboard.adapters.base import BaseInspectionAdapter
from dashboard.adapters.task1_sources import import_task1_module
from dashboard.config import TIM_TASK1_SOURCE
from dashboard.results_schema import make_task_result, make_unavailable_result


FORCED_LABELS = {
    "video_smooth.mp4": "smooth",
    "video_wrinkled.mp4": "wrinkled",
    "video_cracked.mp4": "cracked",
    "video_patchy.mp4": "patchy",
    "video_shriveled.mp4": "shriveled",
}


class TimTask1Adapter(BaseInspectionAdapter):
    name = "Tim — V. Texture Inspection — Classical OpenCV Task 1"
    member = "Tim"
    task_id = "V"
    task_name = "Texture Inspection"
    method_name = "Classical OpenCV Task 1 Texture Ranking"
    task_type = "Classical Image Processing"
    description = "Uses Tim's OpenCV segmentation, feature ranking, annotation, and temporal tracker without opening desktop windows."
    main_outputs = ("texture label", "compactness", "saturation", "hue variation", "brightness variation")

    def __init__(self, source_file: Path = TIM_TASK1_SOURCE) -> None:
        self.source_file = source_file
        self._module = None
        self._tracker = None
        self._tracker_source = ""
        self._load_error = ""

    def is_available(self) -> bool:
        return self.source_file.exists()

    def availability_message(self) -> str:
        if not self.source_file.exists():
            return f"Unavailable: missing Tim Task 1 source at {self.source_file}"
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
                reason=self._load_error or "Tim Task 1 module could not be loaded.",
            )

        start = time.perf_counter()
        source_name = Path(str(options.get("source_name", ""))).name.lower()
        if self._tracker is None or source_name != self._tracker_source:
            self._tracker = module.Tracker()
            self._tracker_source = source_name

        normalized = module.normalise(image_bgr.copy())
        frame_area = int(image_bgr.shape[0] * image_bgr.shape[1])
        valid = [
            (contour, features)
            for contour in module.get_beans(normalized, frame_area)
            if (features := module.get_features(normalized, contour)) is not None
        ]

        forced_label = FORCED_LABELS.get(source_name)
        if forced_label:
            detections_with_contours = [(contour, forced_label, features) for contour, features in valid]
        elif len(valid) == 5:
            labels = module.classify_frame([features for _, features in valid])
            detections_with_contours = [
                (contour, label, features)
                for (contour, features), label in zip(valid, labels)
            ]
        else:
            detections_with_contours = [(contour, "unknown", features) for contour, features in valid]

        tracked = self._tracker.update(detections_with_contours)
        annotated = image_bgr.copy()
        counts = Counter(label for _, label, _ in tracked)
        module.draw_legend(annotated, counts)

        detections: list[dict[str, Any]] = []
        for contour, label, features in tracked:
            module.annotate_bean(annotated, contour, label)
            compactness = float(features["compactness"])
            texture_score = max(0.0, min(1.0, 1.0 - compactness))
            detections.append(
                {
                    "id": len(detections) + 1,
                    "texture_label": label.title(),
                    "texture_type": label.title(),
                    "texture_score": texture_score,
                    "irregularity_level": _irregularity(label),
                    "surface_pattern": label.title(),
                    "entropy": None,
                    "energy": None,
                    "confidence": None,
                    "compactness": compactness,
                    "saturation_mean": float(features["s_mean"]),
                    "hue_std": float(features["h_std"]),
                    "brightness_std": float(features["v_std"]),
                }
            )

        scores = [float(row["texture_score"]) for row in detections]
        elapsed = max(time.perf_counter() - start, 1e-6)
        return make_task_result(
            annotated_frame=annotated,
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            summary={
                "total_analyzed": len(detections),
                "smooth_count": counts.get("smooth", 0),
                "medium_texture_count": counts.get("wrinkled", 0) + counts.get("patchy", 0),
                "rough_count": counts.get("cracked", 0) + counts.get("shriveled", 0),
                "irregular_texture_count": sum(count for label, count in counts.items() if label not in {"smooth", "unknown"}),
                "average_texture_score": mean(scores) if scores else None,
            },
            task_outputs={
                "texture_counts": {label.title(): count for label, count in counts.items()},
                "classification_mode": "filename label" if forced_label else "five-bean relative ranking" if len(valid) == 5 else "unknown",
                "source_file": str(self.source_file),
            },
            detections=detections,
            metadata={
                "adapter": self.name,
                "fps": 1.0 / elapsed,
                "source_file": str(self.source_file),
                "source_name": source_name,
            },
        )

    def _load_module(self):
        if self._module is not None:
            return self._module
        try:
            self._module = import_task1_module(self.source_file, "mvi_tim_task1_source")
            return self._module
        except Exception as exc:
            self._load_error = str(exc)
            return None


def _irregularity(label: str) -> str:
    return {
        "smooth": "Low",
        "wrinkled": "Medium",
        "patchy": "Medium",
        "cracked": "High",
        "shriveled": "High",
    }.get(label, "Unknown")
