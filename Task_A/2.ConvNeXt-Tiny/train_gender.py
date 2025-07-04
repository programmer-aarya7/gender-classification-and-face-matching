import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np

import config
from dataset import GenderDataset
from model import GenderClassifier
from utils import save_checkpoint, get_metrics

def train_one_epoch(loader, model, optimizer, loss_fn, device, scheduler=None):
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0

    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        loss = loss_fn(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
        lr = optimizer.param_groups[0]['lr']
        loop.set_postfix(loss=loss.item(), lr=lr)
        
    return running_loss / len(loader)

def validate(loader, model, loss_fn, device):
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating")
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss = loss_fn(scores, targets)
            running_loss += loss.item()
            _, preds = torch.max(scores, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    metrics = get_metrics(np.array(all_targets), np.array(all_preds))
    val_loss = running_loss / len(loader)
    return val_loss, metrics

def main():
    print(f"Using device: {config.DEVICE}")

    # --- Augmentations (Slightly reduced aggression) ---
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.4),
        ], p=0.8), # Reduced probability from 0.9 to 0.8
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- Datasets and DataLoaders ---
    train_dataset = GenderDataset(image_dir=config.GENDER_TRAIN_DIR, class_labels=config.CLASS_LABELS, transform=train_transform)
    val_dataset = GenderDataset(image_dir=config.GENDER_VAL_DIR, class_labels=config.CLASS_LABELS, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # --- Model and Loss ---
    model = GenderClassifier(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES_GENDER, pretrained=config.PRETRAINED).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    print("\n--- STAGE 1: Training Classifier Head ---")
    # Freeze all layers in the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
    
    for epoch in range(config.FINE_TUNE_EPOCHS):
        print(f"\n--- Head Training Epoch {epoch+1}/{config.FINE_TUNE_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE)
        # We can validate here too, just to see progress
        _, metrics = validate(val_loader, model, loss_fn, config.DEVICE)
        print(f"Head Training Val Accuracy: {metrics['accuracy']:.4f}")

    print("\n--- STAGE 2: Full Model Fine-Tuning ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_FULL, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LR_FULL, steps_per_epoch=len(train_loader), epochs=config.FULL_TRAIN_EPOCHS)

    best_val_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(config.FULL_TRAIN_EPOCHS):
        print(f"\n--- Full-Tuning Epoch {epoch+1}/{config.FULL_TRAIN_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE, scheduler)
        val_loss, metrics = validate(val_loader, model, loss_fn, config.DEVICE)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1-Score: {metrics['f1_score']:.4f}")

        current_accuracy = metrics['accuracy']
        if current_accuracy > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {current_accuracy:.4f}.")
            best_val_accuracy = current_accuracy
            epochs_no_improve = 0
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename=config.CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve. Counter: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to {config.CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()