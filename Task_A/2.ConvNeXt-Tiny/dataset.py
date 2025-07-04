import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class GenderDataset(Dataset):
    
    def __init__(self, image_dir, class_labels, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.class_labels = class_labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_labels)}
        
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.image_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for fname in sorted(os.listdir(target_dir)):
                path = os.path.join(target_dir, fname)
                item = (path, class_index)
                samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get path and label from our samples list
        path, label = self.samples[idx]
        
        # Load image using OpenCV
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations from Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
            return image, torch.tensor(label, dtype=torch.long), path
    

