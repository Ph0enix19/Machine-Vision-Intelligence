# Ali Dashboard Remediation Report

Date: 2026-06-09

## Outcome

- `Ali — III. Seed Growth Measurement` is available and runs successfully.
- Ali Task 1 now normalizes live/upload frames to its original `1280x1024`
  calibration resolution, so small Streamlit frames are not rejected by the
  original fixed pixel-area thresholds.
- Ali Task 1 draws readable `MATURE`/`IMMATURE` tags and counts on the original
  input resolution.
- Ali's configured weights exist and load as an Ultralytics segmentation model.
- The checkpoint's original class tags are exactly `IMMATURE` and `MATURE`.
- The dashboard now preserves those tags in annotations, detection rows, CSV output,
  class counts, and metric cards.
- Ali Task 2 now reproduces the original colored mask fill, outline, per-seed tag,
  and `MATURE`/`IMMATURE`/`TOTAL` count panel.
- Ali's original files were not modified:
  - `group_members/ali/original/live_seed_classifier.py`
  - `group_members/ali/original/best.pt`
  - `MVI_Task1/Ali_Task1.py`
- Streamlit is running at `http://localhost:8501`.
- Streamlit health check: `200 ok`.

## Environment Preparation

Command:

```powershell
& .\.venv\Scripts\Activate.ps1
python -c "import sys,platform; print(sys.executable, platform.python_version())"
pip --version
```

Output:

```text
C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO\.venv\Scripts\python.exe 3.13.2
pip 24.3.1 from C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO\.venv\Lib\site-packages\pip (python 3.13)
```

Installed package versions:

```text
ultralytics=8.4.53
torch=2.12.0
torchvision=0.27.0
streamlit=1.58.0
opencv-python=4.13.0.92
opencv-python-headless=NOT INSTALLED
numpy=2.4.6
```

`cv2` imports successfully from the installed `opencv-python` distribution, so no
OpenCV installation change was required.

## Adapter Availability

Command:

```powershell
python check_adapters.py
```

Output summary:

```text
Dashboard root: C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO
Configured ALI_YOLO_WEIGHTS: C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO\group_members\ali\original\best.pt
Ali weights exist: True

Hemdan Task 1: available
Adonai Task 1: available
Ali Task 1: available
Hany Task 1: available
Tim Task 1: available
Hemdan YOLO: available
Adonai YOLO: available
Ali YOLO: available
Hany Task 2: available
Tim YOLO: available
```

No `pip install ultralytics` command was needed because Ultralytics was already
installed in the repository virtualenv.

If it is missing on another machine, run:

```powershell
python -m pip install ultralytics
```

## Ali Weights

Configured in `dashboard/config.py`:

```text
C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO\group_members\ali\original\best.pt
```

Checkpoint metadata:

```text
Size: 23,837,812 bytes
SHA256: E2D1F5A57BD9F0D881AEE5FDF8FE05A6EFCE346464FC36E9F69F83659FBBB6A4
Ultralytics task: segment
Model names: {0: 'IMMATURE', 1: 'MATURE'}
```

Other `.pt` candidates found include:

```text
yolo11n-seg.pt
runs/segment/bean_seg_v1/weights/best.pt
runs/segment/bean_seg_v1/weights/last.pt
group_members/adonai/original/weights/best.pt
group_members/adonai/original/weights/last.pt
group_members/tim/original/exp-2.pt
```

No weight copy or replacement was necessary.

## Changes Made

- `dashboard/adapters/ali_adapter.py`
  - Preserves checkpoint class names without renaming them.
  - Adds original `MATURE`/`IMMATURE` tags and confidence to detection records.
  - Uses Ali's original green/red tag colors.
  - Adds exact class counts to summaries and task outputs.
- `dashboard/adapters/ali_task1_adapter.py`
  - Preserves Ali Task 1 uppercase classification tags in result rows.
- `dashboard/results_schema.py`
  - Exposes Ali class tag, confidence, and rule detail in Task III tables/CSV.
  - Preserves class counts when video results are summarized.
- `app.py`
  - Displays `MATURE` and `IMMATURE` metric cards for Task III.
- `check_adapters.py`
  - Prints adapter name, availability, message, and configured weights.
- `test_ali_adapter.py`
  - Loads a local frame, runs Ali YOLO, and verifies original tags.
- `test_dashboard_pages.py`
  - Verifies Ali YOLO is selectable and available on Live, Image, and Video.

## Ali Inference Validation

Command:

```powershell
python test_ali_adapter.py
```

Important output:

```text
Input: C:\Users\Mohmed\Downloads\MVI\output_videos\ali test_classified.mp4
Frame shape: (1024, 1280, 3)
Adapter: Ali — III. Seed Growth Measurement
Available: True
Availability message: Available
total_measured: 8
mature_count: 8
immature_count: 0
class_counts: {'MATURE': 8}
First detection class_name: MATURE
First detection confidence: 0.9611599445343018
Ali class tags returned: ['MATURE']
```

The sample frame happened to contain only detections classified as `MATURE`.
The adapter accepts both checkpoint tags and rejects unexpected replacement tags.

## Dashboard Page Validation

Command:

```powershell
python test_dashboard_pages.py
```

Output:

```text
Live Camera Inspection: PASS
  selected stage: Task 2 - AI / YOLO
  selected module: Ali — III. Seed Growth Measurement
Upload Image: PASS
  selected stage: Task 2 - AI / YOLO
  selected module: Ali — III. Seed Growth Measurement
Upload Video: PASS
  selected stage: Task 2 - AI / YOLO
  selected module: Ali — III. Seed Growth Measurement
```

The Live page is tested in HIK mode because `streamlit-webrtc` requires a real
browser session and cannot initialize inside Streamlit's `AppTest` mock runtime.
The HIK page itself renders cleanly and reports that the vendor SDK is not
installed at:

```text
C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport
```

This does not affect Image, Video, or Browser Webcam mode on a real browser.

## Streamlit Startup

Command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8501
```

Startup stdout:

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Startup stderr:

```text
2026-06-09 02:46:14.130 Uvicorn server started on 0.0.0.0:8501
```

Health command:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501/_stcore/health
```

Health output:

```text
200 ok
```

## Errors Encountered

1. The managed Windows shell sandbox initially returned:

   ```text
   windows sandbox: spawn setup refresh
   ```

   Repository commands were rerun through the approved local shell path.

2. `pip show` produced a Windows `cp1252` `UnicodeEncodeError` while printing
   Ultralytics package metadata containing a non-ASCII character. Package
   versions were collected successfully with `importlib.metadata` instead.

3. Browser-webcam rendering under `AppTest` produced:

   ```text
   AttributeError: Mock object has no attribute '_session_mgr'
   ```

   This is a `streamlit-webrtc` test-harness limitation, not a normal dashboard
   runtime failure. Live-page validation used HIK mode, while the real Streamlit
   server started cleanly.

## Commands To Run Locally

```powershell
cd C:\Users\Mohmed\Downloads\MVI\MVI_Task2_Local_YOLO
& .\.venv\Scripts\Activate.ps1
python check_adapters.py
python test_ali_adapter.py
python test_dashboard_pages.py
streamlit run app.py
```

In the dashboard:

1. Open Live Camera, Upload Image, or Upload Video.
2. Choose `Task 2 - AI / YOLO`.
3. Choose `Ali — III. Seed Growth Measurement`.
4. Ali's displayed class tags remain `MATURE` and `IMMATURE`.

## Full Helper Script: check_adapters.py

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dashboard.adapters import get_adapters
from dashboard.config import ALI_YOLO_WEIGHTS


def adapter_weights(adapter: object) -> str:
    path = getattr(adapter, "weights_path", None)
    if path is not None:
        return str(path)
    if getattr(adapter, "member", "") == "Ali" and "YOLO" in getattr(adapter, "method_name", ""):
        return str(ALI_YOLO_WEIGHTS)
    return "N/A"


def main() -> int:
    print(f"Dashboard root: {ROOT}")
    print(f"Configured ALI_YOLO_WEIGHTS: {ALI_YOLO_WEIGHTS}")
    print(f"Ali weights exist: {ALI_YOLO_WEIGHTS.exists()}")
    print()

    failed = False
    for adapter in get_adapters():
        try:
            available = adapter.is_available()
            print(f"name: {adapter.name}")
            print(f"available: {available}")
            print(f"message: {adapter.availability_message()}")
            print(f"weights: {adapter_weights(adapter)}")
            print()
            failed = failed or not available
        except Exception as exc:
            failed = True
            print(f"name: {getattr(adapter, 'name', type(adapter).__name__)}")
            print("available: ERROR")
            print(f"message: {type(exc).__name__}: {exc}")
            print(f"weights: {adapter_weights(adapter)}")
            print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Full Helper Script: test_ali_adapter.py

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from dashboard.adapters.ali_adapter import AliAdapter


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT.parent / "output_videos" / "ali test_classified.mp4"


def load_test_frame(path: Path):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        return frame

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read the first video frame: {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ali's dashboard YOLO adapter on one local frame.")
    parser.add_argument("media", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    media_path = args.media.resolve()
    print(f"Input: {media_path}")
    frame = load_test_frame(media_path)
    print(f"Frame shape: {frame.shape}")

    adapter = AliAdapter()
    print(f"Adapter: {adapter.name}")
    print(f"Available: {adapter.is_available()}")
    print(f"Availability message: {adapter.availability_message()}")
    print(f"Weights: {adapter.weights_path}")

    result = adapter.process_image(
        frame,
        confidence=args.confidence,
        img_size=args.img_size,
        device=args.device,
    )
    print(f"Result keys: {sorted(result.keys())}")
    print("Summary:")
    print(json.dumps(result.get("summary", {}), indent=2))
    print("Task outputs:")
    print(json.dumps(result.get("task_outputs", {}), indent=2))
    print("First detection:")
    first = (result.get("detections") or [None])[0]
    print(json.dumps(first, indent=2))

    error = result.get("metadata", {}).get("error")
    if error:
        print(f"ERROR: {error}")
        return 1

    labels = {
        detection.get("class_name")
        for detection in result.get("detections", [])
        if detection.get("class_name")
    }
    unexpected = labels.difference({"MATURE", "IMMATURE"})
    if unexpected:
        print(f"ERROR: unexpected Ali class tags: {sorted(unexpected)}")
        return 1
    print(f"Ali class tags returned: {sorted(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Full Helper Script: test_dashboard_pages.py

```python
from __future__ import annotations

from typing import Any, Iterable

from streamlit.testing.v1 import AppTest


ALI_ADAPTER = "Ali — III. Seed Growth Measurement"
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

    by_label(at.selectbox, "Assignment task").set_value(TASK2_STAGE)
    at.run()
    assert not exceptions(at), exceptions(at)

    module = by_label(at.selectbox, module_label)
    assert ALI_ADAPTER in module.options, module.options
    module.set_value(ALI_ADAPTER)
    at.run()
    assert not exceptions(at), exceptions(at)

    titles = [item.value for item in at.title]
    assert expected_title in titles, titles
    assert any(item.value == "Available" for item in at.success), [
        item.value for item in at.success
    ]

    print(f"{page}: PASS")
    print(f"  selected stage: {by_label(at.selectbox, 'Assignment task').value}")
    print(f"  selected module: {by_label(at.selectbox, module_label).value}")
    print(f"  titles: {titles}")


def main() -> int:
    check_page("Live Camera Inspection", "Live module", "Live Camera Inspection")
    check_page("Upload Image", "Image module", "Upload Image Inspection")
    check_page("Upload Video", "Video module", "Upload Video Inspection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
