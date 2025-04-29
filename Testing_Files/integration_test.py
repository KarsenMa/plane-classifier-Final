import random
import os
import csv
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def test_models():
    # Load ground truth
    gt_path = Path("Testing_Files/ground_truth.csv")
    ground_truth = {}
    with open(gt_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ground_truth[row['filename']] = row['class']

    # Get all image and video files
    photo_dir = Path("Testing_Files/Photos")
    video_dir = Path("Testing_Files/Videos")
    all_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        all_files.extend(list(photo_dir.rglob(ext)))
    for ext in ('*.mp4', '*.avi', '*.mov', '*.wmv'):
        all_files.extend(list(video_dir.rglob(ext)))

    # Randomly select up to 10 files
    sampled_files = random.sample(all_files, min(10, len(all_files)))

    # Load models
    custom_model = YOLO("custom.pt")
    fgvc_model = YOLO("fgvc.pt")

    correct = 0
    total = 0
    with open('test_results.txt', 'w') as f:
        f.write("Integration Test Results\n======================\n\n")
        for file_path in sampled_files:
            fname = file_path.name
            gt_class = ground_truth.get(fname, "")
            f.write(f"\nTesting file: {fname}\n-----------------------\n")
            try:
                # Classification prediction
                fgvc_results = fgvc_model.predict(str(file_path), verbose=False)[0]
                if fgvc_results and fgvc_results.probs is not None:
                    pred_id = int(fgvc_results.probs.top1)
                    pred_class = fgvc_model.names[pred_id]
                    conf = float(fgvc_results.probs.data[pred_id].item())
                    f.write(f"Predicted: {pred_class} ({conf*100:.1f}%) | Ground Truth: {gt_class}\n")
                    if gt_class and pred_class == gt_class:
                        print("Correct")
                        correct += 1
                    elif gt_class and pred_class[:4] == gt_class[:4]:
                        print("Correct Class, missed subclass")
                        correct += 1
                else:
                    pred_class = ""
                    f.write("No classification result\n")
                total += 1 if gt_class else 0
            except Exception as e:
                f.write(f"Error processing file: {str(e)}\n")
        # Print summary
        accuracy = (correct / total) * 100 if total else 0
        f.write(f"\nSuccess Rate: {correct}/{total} ({accuracy:.2f}%)\n")
        print(f"Success Rate: {correct}/{total} ({accuracy:.2f}%)")

if __name__ == "__main__":
    test_models()