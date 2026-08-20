"""DermaGuard AI Streamlit application."""

import os
from datetime import datetime, timezone

import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model_loader import DEFAULT_MODEL_PATH, load_model
from predict import predict_image
from backend.ai.gradcam import GradCAM, preprocess_image, render_gradcam
from utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    disease_info,
    display_name,
    load_image,
)


st.set_page_config(
    page_title="DermaGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
    :root { --ink:#17324d; --muted:#60768a; --teal:#087f8c; --mint:#e9f7f4; --line:#d8e6eb; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: 0; }
    .stApp { background: linear-gradient(135deg, #f5fbfc 0%, #ffffff 48%, #f1f7fa 100%); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #17324d; }
    [data-testid="stSidebar"] * { color: #eaf5f7 !important; }
    .brand { border-bottom: 1px solid var(--line); padding: 1.4rem 0 1.1rem; margin-bottom: 2rem; }
    .eyebrow { color: var(--teal); font-size: .77rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { padding: 1rem 0 1.5rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 650px; }
    .result-box { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 1.3rem 1.4rem; box-shadow: 0 12px 32px rgba(23,50,77,.07); }
    .result-label { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .result-value { color: var(--ink); font: 800 1.8rem 'Manrope', sans-serif; margin-top: .35rem; }
    .confidence { color: var(--teal); font-size: 1.15rem; font-weight: 700; }
    .notice { background: var(--mint); border-left: 4px solid var(--teal); color: var(--ink); padding: .9rem 1rem; border-radius: 4px; }
    .footer-note { color: var(--muted); font-size: .82rem; border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='brand'><div class='eyebrow'>Clinical AI</div><h2 style='color:#eaf5f7'>DermaGuard</h2></div>", unsafe_allow_html=True)
    st.markdown("### About this tool")
    st.write("An image classification assistant for three broad skin lesion categories.")
    st.markdown("### Model status")
    configured_model = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    st.caption(f"Checkpoint: `{configured_model}`")
    confidence_threshold = st.slider(
        "Low-confidence warning threshold",
        min_value=50.0,
        max_value=95.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=5.0,
        format="%.0f%%",
    )
    st.info("For research and screening support only. This tool does not provide a medical diagnosis.")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.markdown("<div class='hero'><div class='eyebrow'>Skin lesion assessment</div><h1>Understand an image in seconds.</h1><p>Upload a clear skin image to receive the model's three most likely categories and their confidence scores.</p></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG, or PNG. Maximum size: 10 MB.",
)

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file)
        preview, action = st.columns([1.15, 1], gap="large")
        with preview:
            st.image(image, caption="Image ready for analysis", use_container_width=True)
        with action:
            st.markdown("### Ready when you are")
            st.write("The image will be resized and normalized to match the model's training pipeline.")
            analyze = st.button("Analyze image", type="primary", use_container_width=True)

        if analyze:
            try:
                with st.spinner("Analyzing image..."):

                    print("STEP 1: Loading model", flush=True)
                    model, device = load_model(configured_model)

                    print("STEP 2: Running prediction", flush=True)
                    predictions = predict_image(image, model, device)

                    heatmap_image = None
                    overlay_image = None

                    try:
                        print("STEP 3: Preprocessing image", flush=True)
                        input_tensor = preprocess_image(image).to(device)

                        print("STEP 4: Starting GradCAM", flush=True)

                        print("BEFORE GRADCAM OBJECT", flush=True)

                        with GradCAM(model, model.layer4[-1]) as gradcam:
                            heatmap, _, _ = gradcam(input_tensor)

                            print("HEATMAP GENERATED", flush=True)

                            heatmap_image, overlay_image = render_gradcam(
                                image,
                                heatmap
                            )

                        print("STEP 5: Rendering heatmap", flush=True)

                        print("STEP 6: GradCAM complete", flush=True)

                    except Exception as e:
                        logger.exception("GradCAM failed")
                        st.warning(f"Grad-CAM visualization unavailable: {e}")
                        heatmap_image = None
                        overlay_image = None


                primary = predictions[0]
                st.session_state.prediction_history.insert(0, {
                    "time": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
                    "file": uploaded_file.name,
                    "prediction": primary["label"],
                    "confidence": primary["confidence"],
                })
                st.session_state.prediction_history = st.session_state.prediction_history[:20]
                st.markdown("## Assessment")
                st.markdown("<div class='notice'>These results are model estimates, not a diagnosis. Consult a qualified clinician for any concerning or changing lesion.</div>", unsafe_allow_html=True)
                st.write("")
                first, second = st.columns(2)
                with first:
                    st.markdown(f"<div class='result-box'><div class='result-label'>Predicted disease</div><div class='result-value'>{display_name(primary['label'])}</div></div>", unsafe_allow_html=True)
                with second:
                    st.markdown(f"<div class='result-box'><div class='result-label'>Confidence score</div><div class='result-value confidence'>{primary['confidence']:.2f}%</div></div>", unsafe_allow_html=True)
                if primary["confidence"] < confidence_threshold:
                    st.warning(
                        f"Low confidence: {primary['confidence']:.2f}% is below your "
                        f"{confidence_threshold:.0f}% threshold. Do not rely on this result; seek professional review."
                    )
                info = disease_info(primary["label"])
                with st.expander(f"About {display_name(primary['label'])}", expanded=True):
                    st.write(info["summary"])
                    st.caption(info["guidance"])
                st.markdown("### Top 3 predictions")
                for rank, item in enumerate(predictions, 1):
                    st.progress(item["confidence"] / 100, text=f"{rank}. {display_name(item['label'])}  |  {item['confidence']:.2f}%")
                if heatmap_image is not None and overlay_image is not None:
                    st.markdown("## Explainability (Grad-CAM)")
                    original_column, heatmap_column, overlay_column = st.columns(3)
                    with original_column:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with heatmap_column:
                        st.image(heatmap_image, caption="Heatmap", use_container_width=True)
                    with overlay_column:
                        st.image(overlay_image, caption="Overlay", use_container_width=True)

                    st.markdown("### Grad-CAM Explanation")

                    if primary["label"] == "malignant":
                        explanation = (
                            "The highlighted red and yellow regions indicate the areas that most influenced "
                            "the model's malignant prediction. Concentrated attention on an irregular lesion "
                            "may suggest suspicious visual patterns, but this is not a medical diagnosis."
                            )

                    elif primary["label"] == "benign":
                        explanation = (
                            "The highlighted regions show where the model focused when predicting a benign lesion. "
                            "The model found visual patterns that are more consistent with benign skin findings."
                        )

                    else:
                        explanation = (
                            "Grad-CAM highlights the facial regions that contributed most to the Non-Neoplastic prediction. "
                            "Red and yellow areas indicate stronger influence on the model's decision, while blue regions "
                            "had little impact. This visualization helps users understand which image features were used "
                            "by the AI during classification."
                        )

                    st.info(explanation)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("The analysis could not be completed. Check the model checkpoint and try again.")
    except ValueError as exc:
        st.error(str(exc))

with st.sidebar:
    st.markdown("### Prediction history")
    if st.session_state.prediction_history:
        for entry in st.session_state.prediction_history:
            st.markdown(
                f"**{display_name(entry['prediction'])}**  "
                f"\n{entry['confidence']:.1f}% · {entry['time']}  "
                f"\n`{entry['file']}`"
            )
        if st.button("Clear history", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.caption("Your recent analyses will appear here for this session.")

st.markdown("<div class='footer-note'>DermaGuard AI is an assistive research tool. Do not use its output as a substitute for professional medical advice.</div>", unsafe_allow_html=True)
