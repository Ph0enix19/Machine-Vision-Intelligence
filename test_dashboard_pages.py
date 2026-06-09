from __future__ import annotations

from typing import Any, Iterable

from streamlit.testing.v1 import AppTest


ALI_TASK1_ADAPTER = "Ali — III. Seed Growth Measurement — Classical OpenCV Task 1"
ALI_TASK2_ADAPTER = "Ali — III. Seed Growth Measurement"
TASK1_STAGE = "Task 1 - Classical / OpenCV"
TASK2_STAGE = "Task 2 - AI / YOLO"


def by_label(elements: Iterable[Any], label: str) -> Any:
    for element in elements:
        if element.label == label:
            return element
    raise AssertionError(f"Widget not found: {label}")


def exceptions(at: AppTest) -> list[str]:
    return [str(item.value) for item in at.exception]


def check_page(page: str, module_label: str, expected_title: str) -> None:
    at = AppTest.from_file("app.py", default_timeout=45).run()
    if page == "Live Camera Inspection":
        # streamlit-webrtc needs a real browser session and is not supported by AppTest.
        at.session_state["live_camera_source"] = "HIK MVS camera"

    by_label(at.radio, "Page").set_value(page)
    at.run()
    assert not exceptions(at), exceptions(at)

    titles = [item.value for item in at.title]
    assert expected_title in titles, titles

    print(f"{page}: PASS")
    for stage, adapter_name in (
        (TASK1_STAGE, ALI_TASK1_ADAPTER),
        (TASK2_STAGE, ALI_TASK2_ADAPTER),
    ):
        by_label(at.selectbox, "Assignment task").set_value(stage)
        at.run()
        assert not exceptions(at), exceptions(at)

        module = by_label(at.selectbox, module_label)
        assert adapter_name in module.options, module.options
        module.set_value(adapter_name)
        at.run()
        assert not exceptions(at), exceptions(at)
        assert any(item.value.startswith("Available") for item in at.success), [
            item.value for item in at.success
        ]
        print(f"  selected stage: {stage}")
        print(f"  selected module: {adapter_name}")
    print(f"  titles: {titles}")


def main() -> int:
    check_page("Live Camera Inspection", "Live module", "Live Camera Inspection")
    check_page("Upload Image", "Image module", "Upload Image Inspection")
    check_page("Upload Video", "Video module", "Upload Video Inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
