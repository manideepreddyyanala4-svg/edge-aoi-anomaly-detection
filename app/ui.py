import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import numpy as np
from PIL import Image

from src.config import Config
from src.inference import AnomalyDetector
from src.visualization import blend_heatmap_with_image, apply_colormap


st.set_page_config(page_title="Edge AOI Anomaly Detection", layout="wide")

CATEGORIES = ["bottle", "carpet", "grid", "capsule", "transistor"]


@st.cache_resource(show_spinner=True)
def load_model(category: str) -> AnomalyDetector:
    """
    Load the detector for the selected category.
    Uses coreset if available, otherwise baseline.
    """
    config = Config(category=category)

    coreset_path = config.memory_bank_dir / f"{category}_coreset_bank.pt"
    baseline_path = config.memory_bank_dir / f"{category}_full_bank.pt"

    if coreset_path.exists():
        bank_type = "coreset"
    elif baseline_path.exists():
        bank_type = "baseline"
    else:
        raise FileNotFoundError(
            f"No memory bank found for category='{category}'. "
            f"Run build_memory first."
        )

    return AnomalyDetector(config=config, bank_type=bank_type)


st.sidebar.header("Settings")

selected_category = st.sidebar.selectbox(
    "Select Category",
    CATEGORIES,
    index=0,
)

st.sidebar.markdown(f"**Active Category:** `{selected_category}`")

try:
    detector = load_model(selected_category)
    st.sidebar.markdown(f"**Bank Type:** `{detector.bank_type}`")
    st.sidebar.markdown(f"**Threshold:** `{detector.threshold:.4f}`")
except Exception as e:
    st.error(str(e))
    st.stop()


st.title("High-Speed Anomaly Detection (Edge AOI)")
st.markdown("Upload an image to detect defects using a PatchCore-style model.")


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    result = detector.predict(image)

    score = float(result["score"])
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)
    latency = float(result["latency_ms"])
    threshold = float(detector.threshold)

    if selected_category in ["grid", "carpet", "transistor"]:
        adjusted_threshold = threshold * 0.85
    else:
        adjusted_threshold = threshold

    status = "FAIL" if score >= adjusted_threshold else "PASS"

    overlay = blend_heatmap_with_image(image_np, heatmap, alpha=0.45)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, width="stretch")

    with col2:
        st.subheader("Anomaly Heatmap Overlay")
        st.image(overlay, width="stretch")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomaly Score", f"{score:.4f}")
    c2.metric("Status", status)
    c3.metric("Latency (ms)", f"{latency:.2f}")

    with st.expander("Show Raw Heatmap"):
        st.image(apply_colormap(heatmap), width="stretch")

else:
    st.info("Upload an image to begin.")
