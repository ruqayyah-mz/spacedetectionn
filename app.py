# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging
import os

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load YOLO Model ----------------
@st.cache_resource
def load_model():
    logger.info("Loading YOLOv8 model...")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
    model = YOLO(MODEL_PATH)
    logger.info(f"Model Loaded. Classes: {model.names}")
    return model

model = load_model()

# ---------------- Prediction ----------------
def predict_image(image: Image.Image):
    if image is None:
        return None, "No image provided."

    try:
        img = np.array(image)

        # Convert RGB → BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Inference
        results = model(img_bgr, conf=0.25)

        # Get annotated result
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Collect detection details
        boxes = results[0].boxes
        text = f"**Total Detections: {len(boxes)}**\n\n"

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            text += f"{i+1}. **{name}** — {conf:.2%}\n"

        if len(boxes) == 0:
            text = "⚠ No objects detected."

        return annotated_rgb, text

    except Exception as e:
        return None, f"❗ Error: {e}"


# ---------------- UI ----------------
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")

st.title("🦺 Space Station Safety Object Detection")
st.write("Upload an image or capture one — you'll only see the output, not the input preview.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Source")
    mode = st.radio("Choose:", ["Upload Image", "Camera"])

    img = None

    if mode == "Upload Image":
        uploaded = st.file_uploader("Upload file", type=["png","jpg","jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")

    else:
        cam = st.camera_input("Click photo")
        if cam:
            img = Image.open(cam).convert("RGB")

    detect = st.button("🔍 Detect")

with col2:
    st.subheader("Result")
    if detect:
        if img is None:
            st.warning("Upload or capture an image first.")
        else:
            result_img, result_text = predict_image(img)
            st.image(result_img, use_container_width=True)
            st.markdown(result_text)


st.markdown("---")
st.write("### Classes Available")
st.code(", ".join([f"{k}: {v}" for k,v in model.names.items()]))
