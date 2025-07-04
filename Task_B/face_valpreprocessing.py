import os
import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO

IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8n.pt").to(DEVICE)

def extract_largest_face(image_path):
    result = yolo_model(image_path, verbose=False)[0]
    img = cv2.imread(image_path)
    if img is None or result.boxes.shape[0] == 0:
        return None
    boxes = [box.xyxy[0].cpu().numpy() for box in result.boxes]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    face_crop = img[int(y1):int(y2), int(x1):int(x2)]
    return cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

def preprocess_val_faces(input_root, output_root, checkpoint_file):
    os.makedirs(output_root, exist_ok=True)
    done_files = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            done_files = set(json.load(f))

    all_processed = []
    total_saved = 0

    for person_folder in sorted(os.listdir(input_root)):
        person_path = os.path.join(input_root, person_folder)
        if not os.path.isdir(person_path): continue

        person_out_dir = os.path.join(output_root, person_folder)
        distorted_out_dir = os.path.join(person_out_dir, "distorted")

        os.makedirs(person_out_dir, exist_ok=True)
        os.makedirs(distorted_out_dir, exist_ok=True)

        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            if os.path.isdir(img_path): continue
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            rel_path = os.path.relpath(img_path, input_root).replace("\\", "/")
            if rel_path in done_files: continue

            try:
                face = extract_largest_face(img_path)
                if face is not None:
                    cv2.imwrite(os.path.join(person_out_dir, file), face)
                    all_processed.append(rel_path)
                    total_saved += 1
            except Exception as e:
                print(f"❌ Error: {img_path}: {e}")

        distortion_path = os.path.join(person_path, "distortion")
        if os.path.exists(distortion_path):
            for file in os.listdir(distortion_path):
                img_path = os.path.join(distortion_path, file)
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                rel_path = os.path.relpath(img_path, input_root).replace("\\", "/")
                if rel_path in done_files: continue

                try:
                    face = extract_largest_face(img_path)
                    if face is not None:
                        cv2.imwrite(os.path.join(distorted_out_dir, file), face)
                        all_processed.append(rel_path)
                        total_saved += 1
                except Exception as e:
                    print(f"❌ Error: {img_path}: {e}")

        if total_saved % 100 == 0 and all_processed:
            with open(checkpoint_file, "w") as f:
                json.dump(list(done_files.union(all_processed)), f)

    with open(checkpoint_file, "w") as f:
        json.dump(list(done_files.union(all_processed)), f)

    print(f"\n✅ Total saved: {total_saved} face images to: {output_root}")

if __name__ == "__main__":
    input_dir = "Task_B/val"
    output_dir = "Task_B/val_yolo_aligned"
    checkpoint_file = "val_preprocessing_checkpoint.json"
    print(" Preprocessing validation data with YOLOv8...")
    preprocess_val_faces(input_dir, output_dir, checkpoint_file)