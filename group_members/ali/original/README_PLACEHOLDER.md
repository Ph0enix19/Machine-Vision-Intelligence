# Ali Module Files

Ali's submitted files are stored here:

- `best.pt`
- `live_seed_classifier.py`
- `MVI TASK 2 ft. Adonai.v2-updated_seeds.yolov8.zip`

The dashboard does not import `live_seed_classifier.py` directly because it starts
MVS camera and live YOLO work at top level. The safe dashboard adapter is:

`dashboard/adapters/ali_adapter.py`

It exposes Ali as Task III seed growth measurement.

