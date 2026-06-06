# Vision-Based Seed Inspection Dashboard

Streamlit dashboard for an APU Machine Vision assignment. The app combines several seed inspection modules into one interface for live camera input, image upload, video upload, result comparison, and CSV export.

## Features

- Seed classification with YOLO segmentation and OpenCV fallbacks.
- Seed quality inspection for healthy and defective beans.
- Growth measurement outputs such as length, width, area, perimeter, aspect ratio, and circularity.
- Maturity and health analysis using colour features.
- Texture inspection using localization and OpenCV texture metrics.
- Tim Task 1 texture classification using his original OpenCV ranking and temporal tracking logic.
- Upload workflows for images and videos, plus optional local live camera inspection.

## App Entry Point

For Streamlit deployment, use:

```text
app.py
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from this GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

The app includes the required model weight paths used by the dashboard:

```text
runs/segment/bean_seg_v1/weights/best.pt
group_members/ali/original/best.pt
group_members/adonai/original/weights/best.pt
group_members/tim/original/exp-2.pt
```

## Optional Secrets

If the Hany Roboflow adapter is used, add these values in Streamlit secrets or environment variables:

```text
MVI_HANY_ROBOFLOW_API_KEY=your_key_here
MVI_HANY_ROBOFLOW_MODEL_ID=mvi-task-2-dqpn6/2
```

The dashboard still runs without these values, but Hany's Task 2 Roboflow adapter is shown as unavailable. It does not substitute an OpenCV fallback.

Do not commit real API keys to this repository. For Streamlit Community Cloud, add the key in the app settings under Secrets. For local development, set the environment variable in your terminal before starting Streamlit.

For local Streamlit development, you can also create `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example` as the template. The real secrets file is ignored by git.

## Project Structure

```text
app.py                    Streamlit dashboard
dashboard/                Shared dashboard logic, adapters, config, and result schema
group_members/            Integrated member model adapters and required weights
group_members/tim/task1_original/
                           Tim's preserved Task 1 OpenCV source
runs/segment/.../best.pt  Hemdan YOLO segmentation weights
scripts/                  Training, validation, and dataset helper scripts
requirements.txt          Python packages for Streamlit deployment
```

## Notes

- Training is not started from the dashboard. Use the scripts folder for local training or validation.
- Streamlit Cloud does not provide access to your local webcam, so image and video upload workflows are the best deployment targets.
- Streamlit Cloud cannot save files to a user-selected folder on the visitor's computer. Processed videos and CSV files are generated on the server and exposed with download buttons in the browser.
- Generated outputs, raw datasets, cache folders, virtual environments, and large source ZIP files are intentionally ignored from git.
