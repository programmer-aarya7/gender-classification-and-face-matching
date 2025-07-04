import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm

# === Constants ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
data_dir = "1.Efficientnet_cbam/augmented"
model_dir = "1.Efficientnet_cbam/model"
os.makedirs(model_dir, exist_ok=True)

# === CBAM Block ===
class CBAMBlock(nn.Module):
    def __init__(self, channel, ratio=8):
        super(CBAMBlock, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        avg_out = self.shared_mlp(avg_pool.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        max_out = self.shared_mlp(max_pool.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        cbam_out = self.sigmoid(avg_out + max_out)
        return x * cbam_out

# === Model ===
class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetCBAM, self).__init__()
        self.base = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
        self.cbam = CBAMBlock(channel=1792)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1792, num_classes)

    def forward(self, x):
        x = self.base.features(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# === Data Preparation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
print(f"✅ Classes: {class_names}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

targets = [dataset[i][1] for i in train_subset.indices]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_subset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# === Evaluate ===
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    avg_loss = total_loss / len(all_labels)
    return acc, avg_loss, prec, rec, f1

# === Train ===
def train_model():
    model = EfficientNetCBAM(num_classes=2).to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.op
