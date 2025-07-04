import os
import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO

# -------- CONFIG -------- #
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", DEVICE)

# -------- Load YOLOv8 Face Detector -------- #
yolo_model = YOLO("yolov8n.pt").to(DEVICE)

# -------- Face Detection & Resize -------- #
def extract_largest_face(image_path):
    result = yolo_model(image_path, verbose=False)[0]
    img = cv2.imread(image_path)
    if result.boxes.shape[0]:
        boxes = [box.xyxy[0].cpu().numpy() for box in result.boxes]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
        face_crop = img[int(y1):int(y2), int(x1):int(x2)]
        return cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    return None

# -------- Main Preprocessing Function -------- #
def preprocess_faces(input_root, output_root, checkpoint_file):
    os.makedirs(output_root, exist_ok=True)

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            done_files = set(json.load(f))
    else:
        done_files = set()

    all_processed = []
    total_saved = 0

    for person_folder in sorted(os.listdir(input_root)):
        person_path = os.path.join(input_root, person_folder)
        if not os.path.isdir(person_path): continue

        person_out_dir = os.path.join(output_root, person_folder)
        os.makedirs(person_out_dir, exist_ok=True)

        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            if os.path.isdir(img_path) or not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if img_path in done_files: continue

            try:
                face = extract_largest_face(img_path)
                if face is not None:
                    save_path = os.path.join(person_out_dir, f"ref_{file}")
                    cv2.imwrite(save_path, face)
                    all_processed.append(img_path)
                    total_saved += 1
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

        dist_dir = os.path.join(person_path, "distortion")
        if os.path.exists(dist_dir):
            for file in os.listdir(dist_dir):
                img_path = os.path.join(dist_dir, file)
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                if img_path in done_files: continue

                try:
                    face = extract_largest_face(img_path)
                    if face is not None:
                        save_path = os.path.join(person_out_dir, f"dist_{file}")
                        cv2.imwrite(save_path, face)
                        all_processed.append(img_path)
                        total_saved += 1
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")

        if total_saved % 100 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(all_processed, f)

    with open(checkpoint_file, "w") as f:
        json.dump(all_processed, f)

    print(f"\n✅ Total saved: {total_saved} face images to: {output_root}")

if __name__ == "__main__":
    input_dir = "Task_B/train"
    output_dir = "Task_B/train_yolo_aligned"
    checkpoint_file = "train_preprocessing_checkpoint.json"

    print(" Running YOLOv8 preprocessing...")
    preprocess_faces(input_dir, output_dir, checkpoint_file)