import torch
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and optimizer state."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and optimizer state from a checkpoint file."""
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Return the epoch number in case you want to resume training
    epoch = checkpoint.get('epoch', 0)
    print(f"=> Loaded checkpoint from epoch {epoch}")
    return epoch

def get_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# --- NEW ADDITION ---
def de_normalize_image(tensor):
    """Converts a normalized PyTorch tensor back to a displayable OpenCV image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().cpu().numpy()
    image = np.transpose(tensor, (1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_bgr

# ... (other functions remain the same)

def get_metrics(y_true, y_pred, average='binary'):
    """
    Calculates and returns a dictionary of classification metrics.
    'average' can be 'binary' for binary classification or 'macro' for multi-class.
    """
    accuracy = accuracy_score(y_true, y_pred)
    # For Top-1 Accuracy, this is all we need from the spec, but we'll calculate others.
    
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # The problem statement specifically asks for these two for face recognition
    return {
        'top1_accuracy': accuracy,
        'macro_f1_score': f1,
        'macro_precision': precision,
        'macro_recall': recall
    }