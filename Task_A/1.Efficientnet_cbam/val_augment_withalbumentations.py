import os
import cv2
import albumentations as A

# === Paths ===
input_base = "1.Efficientnet_cbam/yolo_val_preprocessed"
output_base = "1.Efficientnet_cbam/augmented_val"
os.makedirs(output_base, exist_ok=True)

# === Albumentations Transform ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.Resize(224, 224),
])

# === Apply Augmentations ===
for gender in ['male', 'female']:
    in_dir = os.path.join(input_base, gender)
    out_dir = os.path.join(output_base, gender)
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(in_dir):
        img_path = os.path.join(in_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Skipped unreadable: {img_path}")
            continue

        for i in range(3):
            augmented = transform(image=image)
            aug_img = augmented["image"]
            aug_img_name = img_name.replace(".jpg", f"_aug{i}.jpg")
            save_path = os.path.join(out_dir, aug_img_name)
            cv2.imwrite(save_path, aug_img)

        print(f"[{gender}] ✅ Augmented: {img_name}")
