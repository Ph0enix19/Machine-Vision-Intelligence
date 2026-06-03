from __future__ import annotations

from dashboard.adapters.base import UnavailableAdapter


class AliPlaceholderAdapter(UnavailableAdapter):
    name = "Ali - Pending Integration"
    task_type = "Pending"
    description = "Ali's module placeholder for Task 3 integration."
    unavailable_reason = (
        "Ali's module is pending. Add his code under group_members/ali/original/ "
        "and update ali_placeholder_adapter.py."
    )

