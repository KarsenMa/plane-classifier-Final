"""
train_models.py

Trains two YOLOv8 models using the Ultralytics API:
1. A subclass model (fine-grained aircraft types)
2. A class model (high-level aircraft families)

Both models are trained using 100 epochs and 640px input resolution.
"""

from ultralytics import YOLO

# Paths to YAML configs
subclass_yaml = "data_subclass.yaml"
class_yaml = "data_class.yaml"

# Base YOLOv8 model (you can change this to yolov8n.pt or yolov8x.pt)
base_model = "yolov8m.pt"

# Training config
epochs = 100
imgsz = 640
batch = 16

print("🔧 Training Subclass Model...")
subclass_model = YOLO(base_model)
subclass_model.train(
    data=subclass_yaml,
    epochs=epochs,
    imgsz=imgsz,
    amp=True,
    batch=batch,
    name="subclass_model"
)

print("✅ Subclass model training complete.\n")

print("🔧 Training Class Model...")
class_model = YOLO(base_model)
class_model.train(
    data=class_yaml,
    epochs=epochs,
    imgsz=imgsz,
    amp=True,
    batch=batch,
    name="class_model"
)

print("✅ Class model training complete.\n")