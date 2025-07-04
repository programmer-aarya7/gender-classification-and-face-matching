import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from inference_utils import load_model, get_embedding, euclidean_distance

VAL_PATH = "Task_B/val_yolo_aligned"
THRESHOLD = 0.6
SAVE_SAMPLES = "sample_predictions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_SAMPLES, exist_ok=True)
model = load_model("checkpoints/best_model.pth")

all_preds, all_labels = [], []
positive_saved, negative_saved = 0, 0
MAX_SAMPLES_PER_CLASS = 10

try:
    font = ImageFont.truetype("arialbd.ttf", 14)
except:
    font = ImageFont.load_default()

all_persons = sorted(os.listdir(VAL_PATH))

for person in tqdm(all_persons):
    person_dir = os.path.join(VAL_PATH, person)
    if not os.path.isdir(person_dir): continue

    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files: continue
    ref_path = os.path.join(person_dir, random.choice(image_files))

    distorted_dir = os.path.join(person_dir, "distorted")
    if not os.path.isdir(distorted_dir): continue

    ref_emb = get_embedding(model, ref_path)

    for distorted_img in os.listdir(distorted_dir):
        if not distorted_img.lower().endswith((".jpg", ".jpeg", ".png")): continue

        distorted_path = os.path.join(distorted_dir, distorted_img)
        distorted_emb = get_embedding(model, distorted_path)
        dist = euclidean_distance(ref_emb, distorted_emb)

        pred = 1 if dist < THRESHOLD else 0
        all_preds.append(pred)
        all_labels.append(1)

        save_sample = False
        if pred == 1 and positive_saved < MAX_SAMPLES_PER_CLASS:
            positive_saved += 1
            save_sample = True
        elif pred == 0 and negative_saved < MAX_SAMPLES_PER_CLASS:
            negative_saved += 1
            save_sample = True

        if save_sample:
            try:
                ref_img = Image.open(ref_path).convert("RGB").resize((112, 112))
                dist_img = Image.open(distorted_path).convert("RGB").resize((112, 112))
                combined = Image.new("RGB", (224, 112))
                combined.paste(ref_img, (0, 0))
                combined.paste(dist_img, (112, 0))
                overlay = Image.new('RGBA', (224, 25), (0, 0, 0, 180))
                combined.paste(overlay, (0, 87), overlay)

                draw = ImageDraw.Draw(combined)
                label_text = f"Positive Match (1) ✅" if pred == 1 else f"Negative Match (0) ❌"
                draw.text((6, 90), f"{label_text} | Dist: {dist:.2f}", fill="white", font=font)
                combined.save(os.path.join(SAVE_SAMPLES, f"{positive_saved+negative_saved}_{person}.jpg"))
            except Exception as e:
                print(f"❌ Failed to save image: {e}")

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

print("\n Evaluation Metrics:")
print(f" Accuracy:  {acc:.4f}")
print(f" Precision: {prec:.4f}")
print(f" Recall:    {rec:.4f}")
print(f" F1 Score:  {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Different", "Same"], yticklabels=["Different", "Same"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_SAMPLES, "confusion_matrix.png"))
plt.show()