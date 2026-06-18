"""Streamlit web application for object detection.

Run with:  uv run streamlit run src/objdetect/app/main.py // VS Code Task -> run:app

The UI lets the user upload an everyday photo, pick a detector, set a
confidence threshold and see the detected objects drawn on the image with a
table and per-class counts.
"""

import streamlit as st
from PIL import Image, ImageOps

from objdetect.app.inference import (
    AVAILABLE_MODELS,
    detections_to_frame,
    load_model,
    run_detection,
    select_detections,
    summarize_counts,
)
from objdetect.visualization import draw_detections


@st.cache_resource(show_spinner="Loading model…")
def _get_model(display_name: str):
    """Cache one instance per model so weights load only once per session."""
    return load_model(display_name)


def _hide_streamlit_chrome() -> None:
    """Hide the Deploy button and ⋮ menu, but keep the sidebar expand control."""
    st.markdown(
        """
        <style>
        [data-testid="stAppDeployButton"] {display: none;}
        #MainMenu {visibility: hidden;}
        [data-testid="stSidebarCollapsedControl"] {visibility: visible !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[str, float]:
    """Draw the Settings panel and return the chosen (model name, threshold)."""
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", list(AVAILABLE_MODELS))
        score_threshold = st.slider(
            "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        st.markdown(
            "**Two-stage** (Faster R-CNN): more accurate, slower.\n\n"
            "**One-stage** (YOLO): faster, lighter."
        )
    return model_name, score_threshold


def main() -> None:
    st.set_page_config(page_title="Object Detection", page_icon="🔍", layout="wide")
    _hide_streamlit_chrome()

    st.title("🔍 Object Detection in Everyday Images")

    model_name, score_threshold = _render_sidebar()

    # --- Upload section ---
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Upload a photo to run detection.")
        return

    # --- Run the selected detector on the uploaded image ---
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    model = _get_model(model_name)

    with st.spinner("Running detection…"):
        detections = run_detection(model, image, score_threshold=score_threshold)

    # --- Show original vs. detections side by side ---
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader(f"Detections — {model.name}")
        detection_slot = st.empty()

    if not detections:
        detection_slot.image(
            draw_detections(image, detections), use_container_width=True
        )
        st.warning("No objects detected above the current confidence threshold.")
        return

    counts = summarize_counts(detections)
    st.success(
        f"Found {len(detections)} object(s): "
        + ", ".join(f"{n}× {label}" for label, n in counts.items())
    )
    st.caption("Select one or more rows to show only those boxes on the photo.")

    # --- Results table; selecting rows filters the boxes drawn above ---
    table = st.dataframe(
        detections_to_frame(detections),
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="detection_table",
    )
    selected_rows = table.get("selection", {}).get("rows", [])
    shown = select_detections(detections, selected_rows)
    detection_slot.image(draw_detections(image, shown), use_container_width=True)


if __name__ == "__main__":
    main()
