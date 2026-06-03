from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from dashboard.results_schema import ensure_result, make_unavailable_result, summarize_result


class BaseInspectionAdapter(ABC):
    name = "Base Adapter"
    member = "Unknown"
    task_id = ""
    task_name = "Unknown Task"
    method_name = "Unknown"
    task_type = "Unknown"
    description = "Common interface for seed inspection modules."
    main_outputs: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return True

    def availability_message(self) -> str:
        return "Available" if self.is_available() else "Unavailable"

    @abstractmethod
    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        raise NotImplementedError

    def process_frame(self, frame_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        return self.process_image(frame_bgr, **options)

    def summarize(self, results: dict[str, Any]) -> dict[str, Any]:
        return summarize_result(ensure_result(results))


class UnavailableAdapter(BaseInspectionAdapter):
    unavailable_reason = "This module is not available."

    def is_available(self) -> bool:
        return False

    def availability_message(self) -> str:
        return self.unavailable_reason

    def process_image(self, image_bgr: np.ndarray, **options: Any) -> dict[str, Any]:
        return make_unavailable_result(
            annotated_frame=image_bgr.copy(),
            member=self.member,
            task_id=self.task_id,
            task_name=self.task_name,
            method=self.method_name,
            reason=self.unavailable_reason,
        )
