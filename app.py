from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from dashboard.adapters import get_adapters
from dashboard.browser_camera import get_rtc_configuration, make_video_frame_callback
from dashboard.config import (
    ALI_YOLO_WEIGHTS,
    DATASET_YAML,
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMG_SIZE,
    HEMDAN_YOLO_WEIGHTS,
    OUTPUT_RESULTS_DIR,
    OUTPUT_VIDEOS_DIR,
    ROOT,
    TIM_YOLO_WEIGHTS,
    ensure_output_dirs,
    find_mvi_task1_files,
)
from dashboard.results_schema import aggregate_results, compact_rows, detections_to_rows, ensure_result
from dashboard.utils import (
    bgr_to_rgb,
    call_adapter,
    detections_dataframe,
    dict_dataframe,
    format_value,
    safe_filename,
    save_uploaded_temp,
    timestamp_slug,
    uploaded_image_to_bgr,
)
from dashboard.video_runner import process_video


st.set_page_config(
    page_title="Vision-Based Seed Inspection",
    layout="wide",
    initial_sidebar_state="expanded",
)


TASK_METRICS = {
    "I": [
        ("Total Beans Detected", "total_detected"),
        ("White Kidney Bean Count", "white_kidney_count"),
        ("Speckled Kidney Bean Count", "speckled_kidney_count"),
        ("Dark Kidney Bean Count", "dark_kidney_count"),
        ("Average Confidence", "average_confidence"),
    ],
    "II": [
        ("Total Seeds Inspected", "total_inspected"),
        ("Healthy Count", "healthy_count"),
        ("Defective Count", "defective_count"),
        ("Crack Count", "crack_count"),
        ("Broken Count", "broken_count"),
        ("Moldy Count", "moldy_count"),
        ("Damaged Count", "damaged_count"),
        ("Unknown Count", "unknown_count"),
    ],
    "III": [
        ("Total Seeds Measured", "total_measured"),
        ("Average Length", "average_length"),
        ("Average Width", "average_width"),
        ("Average Area", "average_area"),
        ("Average Aspect Ratio", "average_aspect_ratio"),
        ("Most Common Shape", "most_common_shape"),
    ],
    "IV": [
        ("Total Seeds Analyzed", "total_analyzed"),
        ("Mature Count", "mature_count"),
        ("Semi-Mature Count", "semi_mature_count"),
        ("Immature Count", "immature_count"),
        ("Overripe Count", "overripe_count"),
        ("Healthy Count", "healthy_count"),
        ("Discoloration Count", "discoloration_count"),
        ("Dark Patch Count", "dark_patch_count"),
    ],
    "V": [
        ("Total Seeds Analyzed", "total_analyzed"),
        ("Smooth Count", "smooth_count"),
        ("Medium Texture Count", "medium_texture_count"),
        ("Rough Count", "rough_count"),
        ("Irregular Texture Count", "irregular_texture_count"),
        ("Average Texture Score", "average_texture_score"),
    ],
}

TASK_STAGE_OPTIONS = {
    "Task 1 - Classical / OpenCV": "task1",
    "Task 2 - AI / YOLO": "task2",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --border: #d8dee9;
            --surface: #fbfcfe;
            --surface-strong: #eef6f1;
            --ink: #111827;
            --muted: #4b5563;
            --accent: #176f4d;
            --accent-2: #0f766e;
            --warn: #9a3412;
            --danger: #991b1b;
        }
        .main .block-container { padding-top: 1.25rem; max-width: 1380px; }
        h1, h2, h3 { letter-spacing: 0; }
        .hero-line { border-left: 4px solid var(--accent); padding-left: 14px; color: var(--muted); margin-bottom: 1rem; }
        .member-card, .status-card {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 212px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .member-card {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .member-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: var(--ink);
        }
        .member-task {
            font-weight: 700;
            color: #1f2937;
            line-height: 1.25;
        }
        .task-badge {
            display: inline-block;
            background: var(--surface-strong);
            color: var(--accent);
            border: 1px solid #b7dec5;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .stage-row {
            border-top: 1px solid #e5e7eb;
            padding-top: 8px;
        }
        .stage-title {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #374151;
            margin-bottom: 4px;
        }
        .method-chip {
            display: inline-block;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            padding: 2px 8px;
            margin: 2px 4px 2px 0;
            background: #ffffff;
            color: #374151;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 0.76rem;
            font-weight: 800;
            margin-left: 4px;
        }
        .status-pill.ok {
            color: #166534;
            background: #dcfce7;
            border: 1px solid #86efac;
        }
        .status-pill.warn {
            color: #92400e;
            background: #fef3c7;
            border: 1px solid #fcd34d;
        }
        .status-ok { color: var(--accent); font-weight: 700; }
        .status-warn { color: var(--warn); font-weight: 700; }
        .muted { color: var(--muted); font-size: 0.92rem; }
        .outputs-line {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .method-strip {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--muted);
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_adapters():
    return get_adapters()


def init_state() -> None:
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("latest_results", [])
    st.session_state.setdefault("latest_detection_rows", [])


def adapter_by_name(name: str):
    return next(adapter for adapter in load_adapters() if adapter.name == name)


def adapter_stage(adapter: Any) -> str:
    source_file = str(getattr(adapter, "source_file", "") or "").replace("\\", "/").lower()
    name = str(getattr(adapter, "name", "")).lower()
    method_name = str(getattr(adapter, "method_name", "")).lower()
    if "mvi_task1" in source_file or "task 1" in name or "task 1" in method_name:
        return "task1"
    return "task2"


def adapters_for_stage(stage: str) -> list[Any]:
    return [adapter for adapter in load_adapters() if adapter_stage(adapter) == stage]


def adapter_selector(label: str, key: str):
    stage_label = st.sidebar.selectbox(
        "Assignment task",
        list(TASK_STAGE_OPTIONS.keys()),
        key=f"{key}_task_stage_label",
    )
    stage = TASK_STAGE_OPTIONS[stage_label]
    st.sidebar.caption(
        "Task 1 shows classical/OpenCV modules. Task 2 shows AI/YOLO neural-network modules."
    )
    adapters = adapters_for_stage(stage)
    if not adapters:
        st.sidebar.warning("No modules are registered for this assignment task.")
        return load_adapters()[0]
    selected_name = st.sidebar.selectbox(label, [adapter.name for adapter in adapters], key=key)
    adapter = adapter_by_name(selected_name)
    st.sidebar.caption(adapter.description)
    if adapter.is_available():
        st.sidebar.success(adapter.availability_message())
    else:
        st.sidebar.warning(adapter.availability_message())
    return adapter


def inference_controls(prefix: str) -> dict[str, Any]:
    confidence = st.sidebar.slider("Confidence", 0.05, 0.95, DEFAULT_CONFIDENCE, 0.05, key=f"{prefix}_confidence")
    img_size = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800, 960], value=DEFAULT_IMG_SIZE, key=f"{prefix}_img_size")
    device = st.sidebar.text_input("Device", value=DEFAULT_DEVICE, placeholder="blank, cpu, or 0", key=f"{prefix}_device")
    return {"confidence": confidence, "img_size": img_size, "device": device}


def remember_run(label: str, result: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> None:
    result = ensure_result(result)
    rows = rows if rows is not None else compact_rows(detections_to_rows(result, source=label))
    st.session_state.latest_results = [result]
    st.session_state.latest_detection_rows = rows
    st.session_state.run_history.append({"label": label, "time": timestamp_slug(), "result": result, "rows": rows})
    st.session_state.run_history = st.session_state.run_history[-12:]


def remember_many(results: list[tuple[str, dict[str, Any]]]) -> None:
    all_rows: list[dict[str, Any]] = []
    st.session_state.latest_results = [ensure_result(result) for _, result in results]
    for label, result in results:
        rows = compact_rows(detections_to_rows(result, source=label))
        all_rows.extend(rows)
        st.session_state.run_history.append({"label": label, "time": timestamp_slug(), "result": result, "rows": rows})
    st.session_state.latest_detection_rows = all_rows
    st.session_state.run_history = st.session_state.run_history[-12:]


def display_result(result: dict[str, Any], title: str | None = None) -> None:
    result = ensure_result(result)
    heading = title or result.get("method", "Inspection Result")
    st.subheader(heading)
    st.markdown(
        f"<div class='method-strip'><span class='task-badge'>Task {result.get('task_id')}</span> "
        f"<strong>{result.get('member')}</strong> - {result.get('task_name')} - {result.get('method')}</div>",
        unsafe_allow_html=True,
    )
    if result.get("metadata", {}).get("error"):
        st.error(result["metadata"]["error"])

    render_metric_cards(result)
    rows_df = detections_dataframe(result, source=result.get("method", ""))
    st.caption("Task-specific detection table")
    if rows_df.empty:
        st.info("No task records were produced for this run.")
    else:
        st.dataframe(rows_df, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            rows_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_filename(result.get('member', 'result'))}_{timestamp_slug()}.csv",
            mime="text/csv",
            key=f"csv_{result.get('member')}_{result.get('method')}_{timestamp_slug()}",
        )

    with st.expander("Technical details", expanded=False):
        outputs = result.get("task_outputs") or {}
        if outputs:
            for key, value in outputs.items():
                st.caption(key.replace("_", " ").title())
                if isinstance(value, dict):
                    st.dataframe(dict_dataframe(value), width="stretch", hide_index=True)
                else:
                    st.write(value)
        else:
            st.write("No extra task outputs.")


def render_metric_cards(result: dict[str, Any]) -> None:
    result = ensure_result(result)
    summary = result.get("summary") or {}
    metric_defs = TASK_METRICS.get(result.get("task_id", ""), [])
    visible_metrics = [(label, key) for label, key in metric_defs if key in summary and summary.get(key) is not None]
    if result.get("metadata", {}).get("fps"):
        visible_metrics.append(("FPS", "__fps"))
    if not visible_metrics:
        visible_metrics = [("Result", "primary_result")]
    cols = st.columns(min(4, max(1, len(visible_metrics))))
    for index, (label, key) in enumerate(visible_metrics):
        value = result["metadata"].get("fps") if key == "__fps" else summary.get(key)
        cols[index % len(cols)].metric(label, format_value(value))


def stage_block(stage_label: str, adapters: list[Any], empty_text: str) -> str:
    if not adapters:
        return (
            "<div class='stage-row'>"
            f"<span class='stage-title'>{html.escape(stage_label)}</span>"
            "<span class='status-pill warn'>Pending</span>"
            f"<div class='outputs-line'>{html.escape(empty_text)}</div>"
            "</div>"
        )
    available = any(adapter.is_available() for adapter in adapters)
    status_class = "ok" if available else "warn"
    status_text = "Available" if available else "Unavailable"
    chips = "".join(
        f"<span class='method-chip'>{html.escape(str(getattr(adapter, 'method_name', adapter.name)))}</span>"
        for adapter in adapters
    )
    sources = [
        Path(str(getattr(adapter, "source_file"))).name
        for adapter in adapters
        if getattr(adapter, "source_file", None)
    ]
    source_line = ""
    if sources:
        source_line = f"<div class='outputs-line'>Source: {html.escape(', '.join(sources))}</div>"
    return (
        "<div class='stage-row'>"
        f"<span class='stage-title'>{html.escape(stage_label)}</span>"
        f"<span class='status-pill {status_class}'>{status_text}</span>"
        f"<div>{chips}</div>"
        f"{source_line}"
        "</div>"
    )


def page_home() -> None:
    st.title("Vision-Based Seed Inspection - Task 3 Group Dashboard")
    st.markdown(
        "<div class='hero-line'>Group GUI integration for image processing methods, Task 1 + Task 2, AI identification, sorting quantities, live video, and task-specific result tables.</div>",
        unsafe_allow_html=True,
    )
    adapters = load_adapters()
    task_cards = [
        ("Hemdan", "I", "Seed Classification", "Classical OpenCV + YOLO Segmentation", "class name, class count, confidence, mask/contour"),
        ("Adonai", "II", "Quality Inspection", "CLAHE + YOLO quality wrapper", "healthy/defective status, crack, broken, moldy, damaged"),
        ("Ali", "III", "Seed Growth Measurement", "YOLO segmentation + pixel measurement", "length, width, area, perimeter, aspect ratio, circularity"),
        ("Hany", "IV", "Maturity & Health", "HSV/RGB color analysis", "RGB, HSV, color uniformity, discoloration, maturity, health"),
        ("Tim", "V", "Texture Inspection", "YOLO localization + texture analysis", "texture label, texture score, surface pattern, irregularity"),
    ]
    st.subheader("Group Member Responsibilities")
    cols = st.columns(3)
    for index, (member, task_id, task_name, _methods, outputs) in enumerate(task_cards):
        member_adapters = [adapter for adapter in adapters if adapter.member == member]
        task1_adapters = [adapter for adapter in member_adapters if adapter_stage(adapter) == "task1"]
        task2_adapters = [adapter for adapter in member_adapters if adapter_stage(adapter) == "task2"]
        with cols[index % 3]:
            st.markdown(
                f"<div class='member-card'>"
                f"<div><span class='task-badge'>Responsibility {html.escape(task_id)}</span>"
                f"<div class='member-title'>{html.escape(member)}</div>"
                f"<div class='member-task'>{html.escape(task_name)}</div></div>"
                f"{stage_block('Task 1', task1_adapters, 'Pending. Add the member Task 1 file under MVI_Task1.')}"
                f"{stage_block('Task 2', task2_adapters, 'Pending. Add the member Task 2 dashboard adapter.')}"
                f"<div class='outputs-line'><strong>Main outputs:</strong> {html.escape(outputs)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Adapter Status")
    st.dataframe(adapter_status_dataframe(), width="stretch", hide_index=True)

    st.subheader("System Checks")
    status_cols = st.columns(4)
    status_cols[0].markdown(status_card("Hemdan YOLO", HEMDAN_YOLO_WEIGHTS.exists(), HEMDAN_YOLO_WEIGHTS), unsafe_allow_html=True)
    status_cols[1].markdown(status_card("Dataset YAML", DATASET_YAML.exists(), DATASET_YAML), unsafe_allow_html=True)
    status_cols[2].markdown(status_card("Ali Weights", ALI_YOLO_WEIGHTS.exists(), ALI_YOLO_WEIGHTS), unsafe_allow_html=True)
    status_cols[3].markdown(status_card("Tim Weights", TIM_YOLO_WEIGHTS.exists(), TIM_YOLO_WEIGHTS), unsafe_allow_html=True)
def page_upload_image() -> None:
    st.title("Upload Image Inspection")
    adapter = adapter_selector("Image module", "image_adapter")
    options = inference_controls("image")
    run_all = st.checkbox("Run all available modules", value=False)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        return

    image_bgr = uploaded_image_to_bgr(uploaded)
    st.image(bgr_to_rgb(image_bgr), channels="RGB", caption="Original image", width="stretch")
    if not st.button("Run inspection", type="primary"):
        return

    selected_stage = TASK_STAGE_OPTIONS.get(st.session_state.get("image_adapter_task_stage_label", ""), "task2")
    adapters_to_run = [item for item in adapters_for_stage(selected_stage) if item.is_available()] if run_all else [adapter]
    results: list[tuple[str, dict[str, Any]]] = []
    for item in adapters_to_run:
        with st.spinner(f"Running {item.name}..."):
            result = call_adapter(item, image_bgr, source_name=uploaded.name, **options)
        results.append((item.name, result))
    remember_many(results)

    if len(results) == 1:
        result = results[0][1]
        annotated = result.get("annotated_frame")
        if annotated is not None:
            st.image(bgr_to_rgb(annotated), channels="RGB", caption="Annotated output", width="stretch")
        display_result(result, title=results[0][0])
        return

    tabs = st.tabs([tab_label(result) for _, result in results])
    for tab, (_, result) in zip(tabs, results):
        with tab:
            annotated = result.get("annotated_frame")
            if annotated is not None:
                st.image(bgr_to_rgb(annotated), channels="RGB", caption="Annotated output", width="stretch")
            display_result(result)


def page_upload_video() -> None:
    st.title("Upload Video Inspection")
    adapter = adapter_selector("Video module", "video_adapter")
    options = inference_controls("video")
    processing_mode = st.radio("Processing mode", ["Process full video", "Limit frames", "Preview only"], horizontal=True)
    max_frames = None
    preview_only = processing_mode == "Preview only"
    if processing_mode == "Limit frames":
        max_frames = st.number_input("Maximum frames", min_value=1, max_value=100000, value=300, step=50)
    elif preview_only:
        max_frames = st.number_input("Preview frame count", min_value=1, max_value=5000, value=60, step=10)
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        return
    if processing_mode == "Process full video":
        st.info("Full video processing can take time for long files. The progress bar will update frame by frame.")
    if not st.button("Process video", type="primary"):
        return

    ensure_output_dirs()
    suffix = Path(uploaded.name).suffix or ".mp4"
    temp_input = save_uploaded_temp(uploaded, suffix)
    stem = safe_filename(Path(uploaded.name).stem)
    slug = timestamp_slug()
    output_video = OUTPUT_VIDEOS_DIR / f"{stem}_{slug}_processed.mp4"
    output_csv = OUTPUT_RESULTS_DIR / f"{stem}_{slug}_detections.csv"
    progress = st.progress(0)
    preview_slot = st.empty()
    status_slot = st.empty()

    def on_progress(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0))
        status_slot.write(f"Processed {done} frame(s)" + (f" of {total}" if total else ""))

    def on_preview(result: dict[str, Any], frame_number: int) -> None:
        annotated = result.get("annotated_frame")
        if annotated is not None:
            preview_slot.image(bgr_to_rgb(annotated), channels="RGB", caption=f"Preview frame {frame_number}", width="stretch")

    try:
        report = process_video(
            temp_input,
            adapter,
            output_video,
            output_csv,
            progress_callback=on_progress,
            preview_callback=on_preview,
            max_frames=int(max_frames) if max_frames else None,
            preview_only=preview_only,
            source_name=uploaded.name,
            **options,
        )
    finally:
        temp_input.unlink(missing_ok=True)

    progress.progress(1.0)
    final_result = report["final_result"]
    if final_result:
        remember_run(f"Video - {adapter.name}", final_result, rows=report.get("latest_rows", []))
        st.info("Video summary below is based on the latest processed frame only. The saved CSV still contains frame-by-frame records.")
        display_result(final_result, title=f"Video Result - {adapter.name}")
    st.success("CSV results are ready.")
    if not preview_only:
        st.success("Processed video is ready.")
        st.video(str(output_video))
        if Path(output_video).exists():
            st.download_button(
                "Download processed video",
                Path(output_video).read_bytes(),
                file_name=output_video.name,
                mime="video/mp4",
            )
    if Path(output_csv).exists():
        st.download_button("Download video CSV", Path(output_csv).read_bytes(), file_name=output_csv.name, mime="text/csv")


def page_live() -> None:
    st.title("Live Camera Inspection")
    adapter = adapter_selector("Live module", "live_adapter")
    options = inference_controls("live")
    if not adapter.is_available():
        st.error(adapter.availability_message())
        return

    webrtc_streamer(
        key=f"browser-camera-{safe_filename(adapter.name)}",
        video_frame_callback=make_video_frame_callback(adapter, options),
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": {"ideal": "environment"},
            },
            "audio": False,
        },
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
        },
    )


def page_results() -> None:
    st.title("Results / Comparison")
    latest_results = st.session_state.get("latest_results", [])
    latest_rows = st.session_state.get("latest_detection_rows", [])
    if not latest_results:
        st.info("Run an image, video, or live inspection first.")
        return
    aggregate = aggregate_results(latest_results)
    col1, col2, col3 = st.columns(3)
    col1.metric("Result Sets", len(latest_results))
    col2.metric("Detection Rows", aggregate["row_count"])
    col3.metric("Tasks Represented", len(aggregate["tasks"]))

    st.subheader("Latest Result Summaries")
    st.dataframe(pd.DataFrame(aggregate["summaries"]), width="stretch", hide_index=True)
    st.subheader("Latest Task-Specific Rows")
    df = pd.DataFrame(latest_rows)
    if df.empty:
        st.info("No detection rows in the latest run.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download combined CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"combined_results_{timestamp_slug()}.csv", mime="text/csv")

    st.subheader("Recent Run History")
    history_rows = []
    for item in st.session_state.get("run_history", []):
        result = ensure_result(item["result"])
        history_rows.append(
            {
                "Time": item["time"],
                "Run": item["label"],
                "Member": result.get("member"),
                "Task": result.get("task_name"),
                "Rows": len(item.get("rows", [])),
                "Primary Result": result.get("summary", {}).get("primary_result", ""),
            }
        )
    st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)


def page_about() -> None:
    st.title("About Methods")
    st.subheader("Task I - Seed Classification")
    st.write("Hemdan's YOLO and classical OpenCV modules identify White, Speckled, and Dark Kidney Beans. Outputs focus on class counts, confidence where available, mask area, and bounding boxes.")
    st.subheader("Task II - Quality Inspection")
    st.write("Adonai's original quality workflow is preserved. The safe wrapper is ready for local weights and reports healthy/defective condition plus crack, broken, moldy, and damaged categories.")
    st.subheader("Task III - Seed Growth Measurement")
    st.write("Ali's provided YOLO segmentation weights are used safely for seed localization. The dashboard computes pixel length, width, area, perimeter, aspect ratio, circularity, compactness, equivalent diameter, and shape.")
    st.subheader("Task IV - Maturity and Health Condition")
    st.write("Hany's safe dashboard module performs RGB/HSV colour analysis, colour uniformity, dark patch ratio, discoloration, maturity labels, and health labels.")
    st.subheader("Task V - Texture Inspection")
    st.write("Tim's YOLO model localizes seeds. OpenCV texture analysis then estimates smooth, medium, or rough surface condition using edge density, entropy, and energy.")
    st.subheader("Assignment Alignment")
    st.write("The dashboard integrates individual image processing programs, overlays Task 1 and Task 2 work with a GUI, supports live video, and tables sorting/classification quantities clearly for group demonstration.")


def page_debug() -> None:
    st.title("System Status / Debug")
    st.subheader("Adapter Status")
    st.dataframe(adapter_status_dataframe(), width="stretch", hide_index=True)
    st.subheader("Detected MVI_Task1 Files")
    task1_files = find_mvi_task1_files()
    if task1_files:
        st.dataframe(pd.DataFrame({"Path": [str(path) for path in task1_files]}), width="stretch", hide_index=True)
    else:
        st.warning("No MVI_Task1 Python files detected.")
    st.subheader("Paths")
    st.dataframe(
        pd.DataFrame(
            [
                {"Name": "Project root", "Path": str(ROOT), "Exists": ROOT.exists()},
                {"Name": "Dataset YAML", "Path": str(DATASET_YAML), "Exists": DATASET_YAML.exists()},
                {"Name": "Hemdan YOLO weights", "Path": str(HEMDAN_YOLO_WEIGHTS), "Exists": HEMDAN_YOLO_WEIGHTS.exists()},
                {"Name": "Ali weights", "Path": str(ALI_YOLO_WEIGHTS), "Exists": ALI_YOLO_WEIGHTS.exists()},
                {"Name": "Tim weights", "Path": str(TIM_YOLO_WEIGHTS), "Exists": TIM_YOLO_WEIGHTS.exists()},
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Latest raw result objects", expanded=False):
        st.write(st.session_state.get("latest_results", []))


def adapter_status_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Member": adapter.member,
                "Task": f"{adapter.task_id} - {adapter.task_name}",
                "Module": adapter.name,
                "Available": adapter.is_available(),
                "Status": adapter.availability_message(),
                "Main Outputs": ", ".join(adapter.main_outputs),
            }
            for adapter in load_adapters()
        ]
    )


def status_card(title: str, ok: bool, path: Path) -> str:
    css = "status-ok" if ok else "status-warn"
    text = "Found" if ok else "Missing"
    return f"<div class='status-card'><strong>{title}</strong><br><span class='{css}'>{text}</span><br><span class='muted'>{path}</span></div>"


def tab_label(result: dict[str, Any]) -> str:
    result = ensure_result(result)
    return f"{result.get('member')} {result.get('task_id')} - {result.get('task_name')}"


def main() -> None:
    ensure_output_dirs()
    init_state()
    inject_css()
    st.sidebar.title("Task 3 Dashboard")
    page = st.sidebar.radio(
        "Page",
        [
            "Home / Overview",
            "Live Camera Inspection",
            "Upload Image",
            "Upload Video",
            "Results / Comparison",
            "About Methods",
            "System Status / Debug",
        ],
    )
    st.sidebar.caption("Training is never run from this dashboard.")

    if page == "Home / Overview":
        page_home()
    elif page == "Live Camera Inspection":
        page_live()
    elif page == "Upload Image":
        page_upload_image()
    elif page == "Upload Video":
        page_upload_video()
    elif page == "Results / Comparison":
        page_results()
    elif page == "About Methods":
        page_about()
    else:
        page_debug()


if __name__ == "__main__":
    main()
