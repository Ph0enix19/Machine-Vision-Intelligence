# APU Machine Vision Task 2: Kidney Bean Segmentation

This project trains and runs a local Ultralytics YOLO instance segmentation model for three kidney bean classes:

- White Kidney Bean
- Speckled Kidney Bean
- Dark Kidney Bean

The original Roboflow ZIP is kept untouched in the parent workspace. This local project uses a copied YOLOv8 segmentation export, split into `train`, `valid`, and `test`, with `task: segment` in `dataset/data.yaml`.

## Setup

Run these commands from `MVI_Task2_Local_YOLO` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_dataset.py
python scripts/train_segmentation.py
python scripts/validate_model.py
python scripts/live_camera_segmentation.py
```

## Check The Dataset

```powershell
python scripts/check_dataset.py
```

The checker prints:

- class names
- train, valid, and test paths
- image and label counts
- missing or empty labels
- invalid class IDs
- labels that look like boxes instead of segmentation polygons

## Train Locally

```powershell
python scripts/train_segmentation.py
```

The script tries `yolo11n-seg.pt` first. If it cannot load, it falls back to `yolov8n-seg.pt`. Training output is saved under:

```text
runs/segment/bean_seg_v1
```

Final weights:

```text
runs/segment/bean_seg_v1/weights/best.pt
runs/segment/bean_seg_v1/weights/last.pt
```

CLI alternative:

```powershell
yolo task=segment mode=train model=yolo11n-seg.pt data=dataset/data.yaml epochs=300 imgsz=640 batch=4 device=0
```

This training configuration requires CUDA device 0.

## Validate

```powershell
python scripts/validate_model.py
```

This loads `runs/segment/bean_seg_v1/weights/best.pt` and prints mask and box metrics when Ultralytics exposes them.

## Live Camera Inference

```powershell
python scripts/live_camera_segmentation.py
```

The live script:

- opens webcam index `0`
- uses `cv2.CAP_DSHOW` on Windows
- requests `1920x1080` at `30 FPS`
- runs YOLO segmentation with `imgsz=640` and `conf=0.35`
- displays masks, boxes, class labels, confidence, and count per class
- quits when you press `q`

You can edit these constants at the top of `scripts/live_camera_segmentation.py`:

```python
MODEL_PATH = ROOT / "runs" / "segment" / "bean_seg_v1" / "weights" / "best.pt"
CAMERA_INDEX = 0
VIDEO_PATH = ""
CONFIDENCE = 0.35
IMG_SIZE = 640
```

To test with a saved video, set `VIDEO_PATH` to a local file path, for example:

```python
VIDEO_PATH = r"C:\Users\Mohmed\Downloads\MVI\videos\test.mp4"
```

## Common Fixes

Webcam not opening:

- Try `CAMERA_INDEX = 1` or `CAMERA_INDEX = 2`.
- Close Teams, Zoom, OBS, browser tabs, or any app using the camera.
- Check Windows camera privacy settings.

CUDA not available:

- The training script will fall back to CPU.
- CPU training is much slower, so reduce `epochs` for a quick test or use a CUDA-enabled machine.

Dark bean predicted as speckled:

- Add more dark bean examples under different lighting.
- Capture dark and speckled beans in the same scene so the model learns the contrast.
- Check that class labels are correct in the Roboflow project.
- Increase confidence only after the model is trained well.

Low FPS:

- Lower `IMG_SIZE` to `480`.
- Lower webcam resolution to `1280x720`.
- Use `yolo11n-seg.pt` or `yolov8n-seg.pt` nano models for speed.
- Use a GPU if available.

Missing labels:

- Run `python scripts/check_dataset.py`.
- Each image should have a matching `.txt` file in the corresponding `labels` folder.
- Empty label files are allowed only for images with no beans, but this project expects bean instances in the frames.
