import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

DATA_PATH = "Task_B/train_yolo_aligned"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 112
EMBEDDING_DIM = 512
BATCH_SIZE = 64
EPOCHS = 10
THRESHOLD = 0.6
NEG_PAIRS_PER_ID = 2
SAVE_PATH = "checkpoints/best_model.pth"
os.makedirs("checkpoints", exist_ok=True)

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomBrightnessContrast(p=0.2),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.2),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.Normalize(),
    ToTensorV2()
])

class SiamesePairDataset(Dataset):
    def __init__(self, root_dir, transform=None, neg_per_id=2):
        self.transform = transform
        self.pairs = []
        self.labels = []
        self.class_to_images = {}
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        for identity in os.listdir(root_dir):
            img_paths = [os.path.join(root_dir, identity, f)
                         for f in os.listdir(os.path.join(root_dir, identity))
                         if f.lower().endswith(self.valid_exts)]
            if len(img_paths) >= 2:
                self.class_to_images[identity] = img_paths

        identities = list(self.class_to_images.keys())

        for identity in identities:
            pos_imgs = self.class_to_images[identity]
            for i in range(len(pos_imgs)):
                for j in range(i + 1, len(pos_imgs)):
                    self.pairs.append((pos_imgs[i], pos_imgs[j]))
                    self.labels.append(1)

            for _ in range(neg_per_id):
                neg_identity = random.choice([id for id in identities if id != identity])
                neg_img = random.choice(self.class_to_images[neg_identity])
                pos_img = random.choice(pos_imgs)
                self.pairs.append((pos_img, neg_img))
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = np.array(Image.open(img1_path).convert("RGB"))
        img2 = np.array(Image.open(img2_path).convert("RGB"))

        if self.transform:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']

        return img1, img2, torch.tensor(label, dtype=torch.float32)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.projector = nn.Linear(1280, embedding_dim)

    def forward(self, x1, x2):
        f1 = self.projector(self.backbone(x1))
        f2 = self.projector(self.backbone(x2))
        return f1, f2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        dists = torch.norm(emb1 - emb2, dim=1)
        pos = labels * dists.pow(2)
        neg = (1 - labels) * torch.clamp(self.margin - dists, min=0).pow(2)
        return (pos + neg).mean()

if __name__ == '__main__':
    dataset = SiamesePairDataset(DATA_PATH, transform=transform, neg_per_id=NEG_PAIRS_PER_ID)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SiameseNetwork().to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_labels, all_preds = [], []

        for x1, x2, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)

            emb1, emb2 = model(x1, x2)
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            dists = torch.norm(emb1 - emb2, dim=1)
            preds = (dists < THRESHOLD).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        avg_loss = total_loss / len(loader)

        print(f"\n✅ Epoch {epoch+1}: Loss={avg_loss:.4f}")
        print(f"🔎 Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("✅ Best model saved.")

    print("\n Training complete. Model saved to:", SAVE_PATH)