
# 🛫 Plane Classifier Web Application

![Python](https://img.shields.io/badge/python-3.1+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-live-brightgreen)


## Overview

**Plane Classifier** is a Streamlit-based web application that showcases two deep learning models for airplane recognition:

- **Custom Detection** model (`custom.pt`): Identifies commercial aircraft via bounding boxes.
- **FGVC Aircraft Classification** model (`fgvc.pt`): Classifies 100+ airplane types from the FGVC dataset.

The app supports both image and video input, offers frame skipping for performance, and includes QR code access for mobile users.

---

## Dependencies

Install required Python libraries using:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `streamlit` – UI framework
- `ultralytics` – YOLOv8 model loading and inference
- `Pillow` – Image processing
- `opencv-python` – Video handling
- `numpy` – Numerical operations
- `qrcode` – QR code generation
- `streamlit-option-menu` – Custom sidebar navigation

---
## Web App Quick Access

Click to Open: (https://plane-classifier-final-hcvahrcjngedtezhz78tcw.streamlit.app/)

OR

Scan to open:

![QR Code](./Plane_Classifier_QR.png)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/KarsenMa/plane-classifier-Final.git
cd plane-classifier-Final
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model files

Download or move the following models to the root directory:
- `custom.pt`
- `fgvc.pt`

---

## Running the App

```bash
streamlit run app.py
```

---

## How to Use

1. Select a model (`Commercial Jets + BB` or `FGVC 100`).
2. Upload an image or a video.
3. For video:
   - Choose frame skip level.
   - Click **"🚀 Process Video"**.
4. View the prediction and confidence.

---

## Project Structure

```
plane-classifier-Final/
├── app.py
├── custom.pt
├── fgvc.pt
├── requirements.txt
├── packages.txt
├── README.md

```

---

## Troubleshooting

- **YOLO loading errors**: Ensure `.pt` model files are correctly placed and match expected architecture.
- **Video playback issues**: Use `.mp4` or `.avi` files under 100MB for best performance.
- **CUDA errors**: Ensure PyTorch matches your CUDA version, or force CPU usage.

---


