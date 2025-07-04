import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from train_efficientnet_cbam import EfficientNetCBAM
from tqdm import tqdm

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "1.Efficientnet_cbam/augmented_val"
model_path = "1.Efficientnet_cbam/model/best_model.pth"
save_folder = "1.Efficientnet_cbam/test_predictions"
img_size = 224
batch_size = 16
os.makedirs(save_folder, exist_ok=True)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Load Dataset ===
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
print("✅ Classes:", class_names)

targets = [sample[1] for sample in dataset.samples]
class_counts = Counter(targets)
print(f" Original class distribution: {dict(zip(class_names, [class_counts[0], class_counts[1]]))}")

weights = [1.0 / class_counts[label] for label in targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

# === Load Model ===
model = EfficientNetCBAM(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded.")

# === Inference ===
y_true, y_pred = [], []
count = 0

print(" Starting Inference...")
for imgs, labels in tqdm(loader):
    imgs_raw = imgs.clone()
    imgs = torch.stack([normalize(img) for img in imgs]).to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

    for i in range(imgs_raw.size(0)):
        if count >= 20:
            break
        img = imgs_raw[i].permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]

        cv2.putText(img_bgr, f"Predicted: {pred_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_bgr, f"True: {true_label}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        file_path = os.path.join(save_folder, f"sample_{count}_pred_{pred_label}_true_{true_label}.jpg")
        cv2.imwrite(file_path, img_bgr)
        count += 1

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("1.Efficientnet_cbam/confusion_matrix.png")
plt.close()

print(f"\n✅ Saved 20 annotated predictions in: '{save_folder}/'")
