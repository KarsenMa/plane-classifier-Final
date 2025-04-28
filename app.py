
# Import libraries
import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import cv2
import os
import numpy as np
import time
import qrcode
from PIL import Image
import base64
import io


# Set page config
st.set_page_config(page_title="Plane Classifier", page_icon="🛫", layout="wide")

# Create QR code
webapp_url = "https://plane-classifier-final-hcvahrcjngedtezhz78tcw.streamlit.app/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=6,
    border=2,
)
qr.add_data(webapp_url)
qr.make(fit=True)

qr_img = qr.make_image(fill_color="red", back_color="white").convert('RGB')
qr_img = qr_img.resize((120, 120), Image.LANCZOS)

# Save QR to base64
buffer = io.BytesIO()
qr_img.save(buffer, format="PNG")
qr_base64 = base64.b64encode(buffer.getvalue()).decode()

# Layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <h1 style="margin: 0; font-size: 48px;">🛫 Plane Classifier</h1>
        </div>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{qr_base64}" width="100" height="100">
            <div style="margin-top: 8px; font-size: 16px; color: gray;">📱 Scan to open the web app</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Model configuration
MODEL_CONFIG = {
    "Commercial Jets + BB": {"path": "custom.pt", "type": "detection"},
    "FGVC 100": {"path": "fgvc.pt", "type": "classification"}
}


@st.cache_resource
def load_model(model_path):
    """Load a YOLO model from the given path with error handling."""
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        st.error(
            "Please ensure the model file exists and dependencies are installed.")
        return None


# Class name mappings for models
COMMERCIAL_CLASS_NAMES = {
    0: "A300", 1: "A310", 2: "A318", 3: "A319", 4: "A320", 5: "A321",
    6: "B737-2", 7: "B737-3", 8: "B737-4", 9: "B737-5", 10: "B737-6", 11: "B737-7",
    12: "B737-8", 13: "B737-9", 14: "B737-8 MAX", 15: "B737-9 MAX",
    16: "B707", 17: "B727", 18: "B747", 19: "B757", 20: "B767", 21: "B777", 22: "B787"
}

# FGVC_CLASS_NAMES dictionary is large and assumed to be unchanged


def classify_image(model, image, model_type, class_names):
    """Run classification or detection on a single image."""
    results = model.predict(image, verbose=False)
    output_image = image.copy()
    label = "No prediction"
    confidence = 0.0

    if model_type == "detection":
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            cls_id = int(boxes.cls[0].item())
            confidence = float(boxes.conf[0].item())
            label = class_names.get(cls_id, f"Unknown ({cls_id})")
            box = boxes.xyxy[0].tolist()

            draw = ImageDraw.Draw(output_image)
            draw.rectangle(box, outline="red", width=5)
            text_position = (box[0], max(0, box[1] - 20))
            draw.text(text_position,
                      f"{label} {confidence*100:.1f}%", fill="red")
        else:
            label = "No plane detected"

    elif model_type == "classification":
        if results and results[0].probs is not None:
            probs = results[0].probs
            cls_id = probs.top1
            confidence = float(probs.top1conf.item())
            label = class_names.get(cls_id, f"Unknown ({cls_id})")
        else:
            label = "Classification failed"

    return label, confidence, output_image


def stream_video(model, video_path, model_type, class_names, frame_skip=1):
    """Process video frame-by-frame with optional frame skipping."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_display = st.empty()
    label_display = st.empty()
    progress = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / max(fps, 1)

    frame_idx = 0
    processed_frames = 0
    last_label_text = "No prediction calculated."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_progress = min(frame_idx / frame_count,
                               1.0) if frame_count > 0 else 0
        progress.progress(current_progress)

        if frame_idx % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            display_img = pil_img.copy()

            label_text = "Processing..."
            results = model.predict(pil_img, verbose=False)

            if model_type == "detection":
                label_text = "No plane detected"
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    cls_id = int(boxes.cls[0].item())
                    confidence = float(boxes.conf[0].item())
                    label = class_names.get(cls_id, f"Unknown ({cls_id})")
                    label_text = f"{label} ({confidence*100:.1f}%)"

                    box = boxes.xyxy[0].tolist()
                    draw = ImageDraw.Draw(display_img)
                    draw.rectangle(box, outline="red", width=5)
                    text_position = (box[0], max(0, box[1] - 20))
                    draw.text(text_position, label_text, fill="red")

            elif model_type == "classification":
                label_text = "Classification failed"
                if results and results[0].probs is not None:
                    probs = results[0].probs
                    cls_id = probs.top1
                    confidence = float(probs.top1conf.item())
                    label = class_names.get(cls_id, f"Unknown ({cls_id})")
                    label_text = f"{label} ({confidence*100:.1f}%)"

            frame_display.image(
                display_img, caption=f"Frame {processed_frames+1}", use_container_width=True)
            label_display.markdown(f"### 🛫 Prediction: **{label_text}**")

            last_label_text = label_text

            time.sleep(max(delay * frame_skip, 0.01))
            processed_frames += 1

        frame_idx += 1

    cap.release()
    progress.empty()
    label_display.markdown(f"### 🛫 Final Prediction: **{last_label_text}**")
    st.success("🟢 Video processing complete!")


model_option_1 = "Commercial Jets + BB"
model_option_2 = "FGVC 100"

selected_model = option_menu(
    menu_title=None,
    options=[model_option_1, model_option_2],
    icons=['bounding-box', 'card-image'],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5px 0px", "background-color": "#1f1f1f"},
        "icon": {"color": "#cccccc", "font-size": "16px"},
        "nav-link": {
            "font-size": "0.95rem",
            "color": "#cccccc",
            "background-color": "transparent",
            "text-align": "center",
            "margin": "0px 6px",
            "padding": "8px 16px",
            "border-radius": "8px",
            "--hover-color": "#333333"
        },
        "nav-link-selected": {
            "background-color": "#404040",
            "color": "#ffffff",
            "font-weight": "normal",
        },
    }
)

if selected_model == model_option_1:
    st.header(model_option_1)
    config = MODEL_CONFIG[model_option_1]
    model = load_model(config["path"])
    model_type = config["type"]
    current_class_names = COMMERCIAL_CLASS_NAMES

elif selected_model == model_option_2:
    st.header(model_option_2)
    config = MODEL_CONFIG[model_option_2]
    model = load_model(config["path"])
    model_type = config["type"]
    current_class_names = FGVC_CLASS_NAMES

if model:
    file_option = st.radio("Select input type:", [
                           "Image", "Video"], key=f"radio_{selected_model}")

    if file_option == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=[
                                         "jpg", "jpeg", "png"], key=f"upload_img_{selected_model}")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner('Classifying image...'):
                label, confidence, output_image = classify_image(
                    model, image, model_type, current_class_names)
            st.image(output_image, caption="Detected Plane",
                     use_container_width=True)
            st.markdown(f"### ✈️ Prediction: **{label}**")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    elif file_option == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type=[
                                          "mp4", "avi", "mov"], key=f"upload_vid_{selected_model}")
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name

            st.markdown("### ⚡ Speed Options")
            frame_skip = st.slider(
                "Frame skip amount (1 = no skip, higher = faster)", 1, 10, 2, key=f"slider_{selected_model}")

            if st.button("🚀 Process Video", key=f"button_vid_{selected_model}"):
                with st.spinner('Processing video... This may take a while.'):
                    stream_video(model, video_path, model_type,
                                 current_class_names, frame_skip=frame_skip)

            if 'video_path' in locals() and os.path.exists(video_path):
                pass
else:
    st.warning(
        f"Model '{selected_model}' could not be loaded. Please check the file path and logs.")
