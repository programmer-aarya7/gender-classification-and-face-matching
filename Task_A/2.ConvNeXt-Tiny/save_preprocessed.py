import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import config
from dataset import GenderDataset

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

def save_augmented_images(num_images_to_save=100): # Increased default to 100
    """
    Loads the training data, applies augmentations, and saves samples
    to disk for visual inspection.
    """
    output_dir = "preprocessed_output_correct" 
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir) # Clean up old results
    os.makedirs(output_dir)
    
    # Use the same train_transform from the training script
    train_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.4),
        ], p=0.8),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = GenderDataset(
        image_dir=config.GENDER_TRAIN_DIR,
        class_labels=config.CLASS_LABELS,
        transform=train_transform
    )
    
    # NOTE: Set shuffle=False here to get a predictable set of images each time.
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    print(f"Saving {num_images_to_save} augmented images to '{output_dir}/'")
    
    saved_count = 0
    for image_tensor, label_tensor, path_tuple in tqdm(loader, total=num_images_to_save):
        if saved_count >= num_images_to_save:
            break
            
        # path_tuple will be a tuple of one path, so get the first element
        path = path_tuple[0]
        label = label_tensor.item()
        class_name = config.CLASS_LABELS[label]
        
        # Create a subdirectory for the class if it doesn't exist
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # De-normalize and get the image
        augmented_image = de_normalize_image(image_tensor.squeeze(0))
        
        # Create a unique filename
        original_filename = os.path.basename(path)
        save_path = os.path.join(class_dir, f"aug_{saved_count}_{original_filename}")
        
        # Save the image
        cv2.imwrite(save_path, augmented_image)
        saved_count += 1
        
    print(f"Done! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    save_augmented_images(num_images_to_save=100)