import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
import torchvision.models.segmentation as models
from PIL import Image

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load DeepLabV3+ with ResNet101 backbone
def load_deeplab():
    model = models.deeplabv3_resnet101(pretrained=True).to(device)
    model.eval()
    return model

# Load models
deeplab_model = load_deeplab()
yolo_model = YOLO("yolov8n.pt")  # Use .pt model for GPU support

# DeepLab input transforms
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Apply segmentation using DeepLab
def apply_deeplab(face_img):
    input_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab_model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    
    # Resize mask to face image size
    mask = cv2.resize(mask, (face_img.shape[1], face_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Apply segmentation mask
    face_img[mask == 0] = 0
    return face_img

# Detect face, crop, apply DeepLab segmentation
def detect_crop_segment(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    results = yolo_model.predict(source=img, device=0 if torch.cuda.is_available() else 'cpu')[0]
    faces = results.boxes.xyxy.cpu().numpy()

    for i, (x1, y1, x2, y2) in enumerate(faces):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face = cv2.resize(face, (256, 256))  # Resize for DeepLab
        segmented_face = apply_deeplab(face)
        segmented_face = cv2.resize(segmented_face, (224, 224))  # Final resize for model input

        new_save_path = save_path.replace(".jpg", f"_{i}.jpg")
        cv2.imwrite(new_save_path, segmented_face)
    return True

# Input/output folders
input_base = r"C:\Users\aarya\WORKK-SLAY IT\Compsy Hackathon\Comys_Hackathon5\Task_A\train"
output_base = r"C:\Users\aarya\WORKK-SLAY IT\Compsy Hackathon\Comys_Hackathon5\Task_A\deeplab_yolo_preprocessed"

os.makedirs(output_base, exist_ok=True)

# Loop through male/female folders
for gender in ['male', 'female']:
    img_dir = os.path.join(input_base, gender)
    save_dir = os.path.join(output_base, gender)
    os.makedirs(save_dir, exist_ok=True)

    for idx, img_name in enumerate(os.listdir(img_dir)):
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
