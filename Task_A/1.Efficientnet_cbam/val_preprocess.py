import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deeplabv3_model import load_deeplab
import torchvision.transforms as T

# === Load Models ===
yolo_model = YOLO("yolov8n.pt")
deeplab_model = load_deeplab()

# === Transform for DeepLab ===
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def apply_deeplab(face_img):
    input_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        output = deeplab_model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    mask = cv2.resize(mask, (face_img.shape[1], face_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    face_img[mask == 0] = 0
    return face_img

def detect_crop_segment(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    results = yolo_model(img)[0]
    faces = results.boxes.xyxy.cpu().numpy()

    for i, (x1, y1, x2, y2) in enumerate(faces):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (256, 256))
        segmented_face = apply_deeplab(face)
        segmented_face = cv2.resize(segmented_face, (224, 224))
        new_save_path = save_path.replace(".jpg", f"_{i}.jpg")
        cv2.imwrite(new_save_path, segmented_face)
    return True

# === Paths ===
input_base = "1.Efficientnet_cbam/val"
output_base = "1.Efficientnet_cbam/yolo_val_preprocessed"
os.makedirs(output_base, exist_ok=True)

for gender in ['male', 'female']:
    img_dir = os.path.join(input_base, gender)
    save_dir = os.path.join(output_base, gender)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        save_path = os.path.join(save_dir, img_name)

        try:
            success = detect_crop_segment(img_path, save_path)
            if success:
                print(f"[{gender}] ✅ {img_name}")
            else:
                print(f"[{gender}] ❌ No face: {img_name}")
        except Exception as e:
            print(f"[{gender}] ❗ Error processing {img_name}: {e}")
