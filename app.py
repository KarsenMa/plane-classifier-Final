# Save this as app.py
# Run with: streamlit run app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import cv2
import os
import numpy as np
import time

# --- Configuration ---
st.set_page_config(page_title="Plane Classifier", page_icon="✈️", layout="wide")

# --- Custom CSS Injection ---
# Aims to replicate the look from the image. Selectors might need verification.
st.markdown("""
<style>
    /* Apply a dark background if not using Streamlit's dark theme */
    /* body { background-color: #1f1f1f !important; } */ /* May conflict with theme */

    /* Style the container for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Spacing between tabs */
        padding-top: 10px; /* Add some space above */
        padding-bottom: 10px; /* Add some space below */
    }

    /* Style individual tab buttons */
    .stTabs [data-baseweb="tab"] {
        height: 40px; /* Tab height */
        white-space: pre-wrap;
        background-color: transparent; /* Unselected tab background */
        border-radius: 8px; /* Rounded corners */
        color: #cccccc; /* Unselected text color */
        padding: 8px 16px; /* Tab padding */
        margin: 0;
        border: none !important; /* Remove default borders */
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Remove the default bottom border highlight */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
         border-bottom: none !important;
         box-shadow: none !important; /* Remove potential focus shadow on selection */
    }

    /* Style the SELECTED tab button */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #404040; /* Selected background color */
        color: #ffffff; /* Selected text color */
    }

    /* Optional: Hover effect for unselected tabs */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #333333;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
MODEL_CONFIG = {
    "Commercial Jets + BB": {"path": "custom.pt", "type": "detection"},
    "FGVC 100": {"path": "fgvc.pt", "type": "classification"}
}

@st.cache_resource
def load_model(model_path):
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        st.error("Please ensure the model file exists at the specified path and necessary dependencies (like PyTorch) are installed.")
        return None

# --- Class Names ---
COMMERCIAL_CLASS_NAMES = { 0: "A300", 1: "A310", 2: "A318", 3: "A319", 4: "A320", 5: "A321", 6: "B737-2", 7: "B737-3", 8: "B737-4", 9: "B737-5", 10: "B737-6", 11: "B737-7", 12: "B737-8", 13: "B737-9", 14: "B737-8 MAX", 15: "B737-9 MAX", 16: "B707", 17: "B727", 18: "B747", 19: "B757", 20: "B767", 21: "B777", 22: "B787"}
FGVC_CLASS_NAMES = {0: 'Boeing 737-700', 1: 'Boeing 737-800', 2: 'Boeing 737-900', 3: 'Boeing 747-100', 4: 'Boeing 747-200', 5: 'Boeing 747-300', 6: 'Boeing 747-400', 7: 'Boeing 757-200', 8: 'Boeing 757-300', 9: 'Boeing 767-200', 10: 'Boeing 767-300', 11: 'Boeing 767-400', 12: 'Boeing 777-200', 13: 'Boeing 777-300', 14: 'Airbus A300', 15: 'Airbus A310', 16: 'Airbus A318', 17: 'Airbus A319', 18: 'Airbus A320', 19: 'Airbus A321', 20: 'Airbus A330-200', 21: 'Airbus A330-300', 22: 'Airbus A340-200', 23: 'Airbus A340-300', 24: 'Airbus A340-500', 25: 'Airbus A340-600', 26: 'Airbus A380', 27: 'McDonnell Douglas MD-11', 28: 'McDonnell Douglas MD-80', 29: 'McDonnell Douglas MD-87', 30: 'McDonnell Douglas MD-90', 31: 'McDonnell Douglas DC-8', 32: 'McDonnell Douglas DC-9-30', 33: 'McDonnell Douglas DC-10', 34: 'Boeing 707', 35: 'Boeing 717', 36: 'Boeing 727', 37: 'Fokker 100', 38: 'Fokker 50', 39: 'Fokker 70', 40: 'Saab 2000', 41: 'Saab 340', 42: 'Embraer ERJ 135', 43: 'Embraer ERJ 140', 44: 'Embraer ERJ 145', 45: 'Embraer E-170', 46: 'Embraer E-190', 47: 'Embraer E-195', 48: 'Bombardier CRJ-100', 49: 'Bombardier CRJ-200', 50: 'Bombardier CRJ-700', 51: 'Bombardier CRJ-900', 52: 'Bombardier Dash 8-100', 53: 'Bombardier Dash 8-300', 54: 'ATR-42', 55: 'ATR-72', 56: 'BAE 146-200', 57: 'BAE 146-300', 58: 'BAE Systems Avro RJ85', 59: 'BAE Systems Avro RJ100', 60: 'BAE Jetstream 31', 61: 'Ilyushin Il-62', 62: 'Ilyushin Il-76', 63: 'Ilyushin Il-96', 64: 'Tupolev Tu-134', 65: 'Tupolev Tu-154', 66: 'Tupolev Tu-204', 67: 'Antonov An-12', 68: 'Antonov An-24', 69: 'Antonov An-26', 70: 'Cessna 172', 71: 'Cessna 208', 72: 'Cessna 525', 73: 'Cessna 560', 74: 'Beechcraft 1900', 75: 'Beechcraft King Air 200', 76: 'Dornier 328', 77: 'Canadair CL-600', 78: 'Gulfstream IV', 79: 'Gulfstream V', 80: 'Dassault Falcon 2000', 81: 'Dassault Falcon 900', 82: 'Learjet 35', 83: 'Learjet 45', 84: 'Lockheed L-1011 TriStar', 85: 'Bombardier Global Express', 86: 'Dassault Falcon 7X', 87: 'C-130 Hercules', 88: 'C-17 Globemaster III', 89: 'C-5 Galaxy', 90: 'KC-135 Stratotanker', 91: 'E-3 Sentry', 92: 'P-3 Orion', 93: 'F/A-18', 94: 'F-16 Fighting Falcon', 95: 'F-15 Eagle', 96: 'A-10 Thunderbolt II', 97: 'Eurofighter Typhoon', 98: 'Dassault Rafale', 99: 'General Dynamics F-111 Aardvark'}

# --- Prediction Functions ---
def classify_image(model, image, model_type, class_names):
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
            draw.text(text_position, f"{label} {confidence*100:.1f}%", fill="red") # Basic font
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
    # (Function content is the same as in your provided script - kept for brevity)
    # ... (paste the full stream_video function here from your script) ...
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_progress = min(frame_idx / frame_count, 1.0) if frame_count > 0 else 0
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

            frame_display.image(display_img, caption=f"Frame {processed_frames+1}", use_container_width=True)
            label_display.markdown(f"### ✈️ Prediction: **{label_text}**")

            # Add a small sleep even if delay is very small to allow UI updates
            time.sleep(max(delay * frame_skip, 0.01))
            processed_frames += 1

        frame_idx += 1

    cap.release()
    progress.empty()
    label_display.markdown("### ✅ Video Processing Complete!")


# --- Streamlit UI ---
st.title("✈️ Plane Classifier")

# Create tabs for model selection (styled by CSS)
tab1_name = "Commercial Jets + BB"
tab2_name = "FGVC 100"
tab1, tab2 = st.tabs([tab1_name, tab2_name])

# --- Tab 1 Logic ---
with tab1:
    st.header(tab1_name)
    model_name = tab1_name
    config = MODEL_CONFIG[model_name]
    model_path = config["path"]
    model_type = config["type"]
    current_class_names = COMMERCIAL_CLASS_NAMES # Use specific name

    model = load_model(model_path)

    if model:
        file_option_t1 = st.radio("Select input type:", ["Image", "Video"], key=f"radio_{model_name}")

        if file_option_t1 == "Image":
            uploaded_file_t1 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=f"upload_img_{model_name}")
            if uploaded_file_t1:
                image = Image.open(uploaded_file_t1).convert("RGB")
                with st.spinner('Classifying image...'):
                    label, confidence, output_image = classify_image(model, image, model_type, current_class_names)
                st.image(output_image, caption="Detected Plane", use_container_width=True)
                st.markdown(f"### ✈️ Prediction: **{label}**")
                st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        elif file_option_t1 == "Video":
            uploaded_video_t1 = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key=f"upload_vid_{model_name}")
            if uploaded_video_t1:
                # Use a context manager for safer temp file handling
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_video_t1.read())
                    video_path = tfile.name

                st.markdown("### ⚡ Speed Options")
                frame_skip_t1 = st.slider("Frame skip amount (1 = no skip, higher = faster)", 1, 10, 2, key=f"slider_{model_name}")

                if st.button("🚀 Process Video", key=f"button_vid_{model_name}"):
                    with st.spinner('Processing video... This may take a while.'):
                        stream_video(model, video_path, model_type, current_class_names, frame_skip=frame_skip_t1)

                # Clean up the temporary file after processing (if button was clicked)
                if 'video_path' in locals() and os.path.exists(video_path):
                     # Could add logic to delete only after processing finishes,
                     # but for simplicity, it's left as is (might delete early if user navigates away)
                     # Consider adding os.unlink(video_path) after the stream_video call if needed.
                     pass
    else:
        st.warning(f"Model '{model_name}' could not be loaded. Please check the path and logs.")


# --- Tab 2 Logic ---
with tab2:
    st.header(tab2_name)
    model_name = tab2_name
    config = MODEL_CONFIG[model_name]
    model_path = config["path"]
    model_type = config["type"]
    current_class_names = FGVC_CLASS_NAMES # Use specific name

    model = load_model(model_path)

    if model:
        file_option_t2 = st.radio("Select input type:", ["Image", "Video"], key=f"radio_{model_name}")

        if file_option_t2 == "Image":
            uploaded_file_t2 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=f"upload_img_{model_name}")
            if uploaded_file_t2:
                image = Image.open(uploaded_file_t2).convert("RGB")
                with st.spinner('Classifying image...'):
                    label, confidence, output_image = classify_image(model, image, model_type, current_class_names)
                st.image(image, caption="Input Image", use_container_width=True) # Show original for classification
                st.markdown(f"### ✈️ Prediction: **{label}**")
                st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        elif file_option_t2 == "Video":
            uploaded_video_t2 = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key=f"upload_vid_{model_name}")
            if uploaded_video_t2:
                 # Use a context manager for safer temp file handling
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_video_t2.read())
                    video_path = tfile.name

                st.markdown("### ⚡ Speed Options")
                frame_skip_t2 = st.slider("Frame skip amount (1 = no skip, higher = faster)", 1, 10, 2, key=f"slider_{model_name}")

                if st.button("🚀 Process Video", key=f"button_vid_{model_name}"):
                    with st.spinner('Processing video... This may take a while.'):
                        stream_video(model, video_path, model_type, current_class_names, frame_skip=frame_skip_t2)

                # Clean up temp file (similar caution as above)
                if 'video_path' in locals() and os.path.exists(video_path):
                     pass
    else:
        st.warning(f"Model '{model_name}' could not be loaded. Please check the path and logs.")


# --- Footer or other elements outside tabs ---
# st.write("App footer or instructions")