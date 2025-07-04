import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import os
import cv2
import config
from dataset import GenderDataset
from model import GenderClassifier
from utils import load_checkpoint, get_metrics, de_normalize_image

def visualize_predictions(model, loader, device, num_images=20):
    """Saves images with predicted and true labels for visual inspection."""
    print(f"\nSaving {num_images} visualization images to 'output_test_visuals/'...")
    output_dir = "output_test_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    images_saved = 0
    with torch.no_grad():
        # Using iter and next to grab a few samples without iterating the whole dataset
        data_iter = iter(loader)
        for i in tqdm(range(num_images)):
            try:
                # Our dataset returns (image, label, path)
                images, labels, paths = next(data_iter)
            except StopIteration:
                break # Stop if we run out of images

            images = images.to(device)
            labels = labels.to(device)
            
            # Get model prediction
            scores = model(images)
            _, predictions = torch.max(scores, 1)
            
            # De-normalize image for saving
            img_to_save = de_normalize_image(images[0])
            
            # Get true and predicted class names
            true_label_name = config.CLASS_LABELS[labels[0].item()]
            pred_label_name = config.CLASS_LABELS[predictions[0].item()]
            
            # Set text color: Green for correct, Red for incorrect
            if pred_label_name == true_label_name:
                text_color = (0, 255, 0) # Green
                result_text = "Correct"
            else:
                text_color = (0, 0, 255) # Red
                result_text = "Incorrect"

            # Add text to image
            cv2.putText(img_to_save, f"True: {true_label_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_to_save, f"Pred: {pred_label_name}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Save the image
            original_filename = os.path.basename(paths[0])
            save_path = os.path.join(output_dir, f"{result_text}_{i}_{original_filename}")
            cv2.imwrite(save_path, img_to_save)
            images_saved += 1
            
    print(f"Done. {images_saved} images saved.")


def main():
    """Main function to run the evaluation."""
    print(f"Using device: {config.DEVICE}")
    
    # --- Data Transformations ---
    # IMPORTANT: Use the same transformations as the validation set during training
    test_transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- Dataset and DataLoader ---
    # We evaluate on the validation set, as the test set is hidden
    test_dataset = GenderDataset(
        image_dir=config.GENDER_VAL_DIR,
        class_labels=config.CLASS_LABELS,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No need to shuffle for testing
        num_workers=config.NUM_WORKERS
    )
    
    # --- Load Model ---
    model = GenderClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES_GENDER,
        pretrained=False # We are loading our own trained weights
    ).to(config.DEVICE)
    
    load_checkpoint(config.CHECKPOINT_PATH, model)
    model.eval() # Set model to evaluation mode

    # --- Run Inference ---
    all_preds = []
    all_targets = []
    loop = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for images, labels, _ in loop:
            images = images.to(config.DEVICE)
            
            # Forward pass
            scores = model(images)
            
            # Get predictions
            _, preds = torch.max(scores, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # --- Calculate and Display Metrics ---
    print("\n--- Final Performance on Validation Set ---")
    metrics = get_metrics(np.array(all_targets), np.array(all_preds))
    
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    
    # --- Visualize some predictions ---
    visualize_predictions(model, test_loader, config.DEVICE)

if __name__ == "__main__":
    main()