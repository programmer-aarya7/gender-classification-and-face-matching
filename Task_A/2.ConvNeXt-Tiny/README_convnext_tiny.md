
# Task A: Robust Gender Classification

This document details the implementation for **Task A: Gender Classification**, where the objective is to classify gender (Male/Female) from facial images, with a strong emphasis on maintaining accuracy under **challenging visual conditions** like blur, fog, and adverse lighting.

## Objective

The goal is to develop a robust deep learning model that accurately predicts gender from face images that may be:
- Degraded by weather conditions (fog, rain).
- Affected by motion blur or poor lighting (over/under-exposed).
- Captured in a variety of real-world, non-ideal scenarios.

## Pipeline Overview

Our approach is an end-to-end, state-of-the-art transfer learning pipeline that directly learns from the provided images without requiring external face detection or segmentation steps.

```
[Raw Images (Male/Female) under Visual Distortions]
    --> [On-the-fly Data Augmentation with Albumentations]
    --> [State-of-the-Art Backbone: ConvNeXt-Tiny (ImageNet-22K)]
    --> [Two-Stage Fine-Tuning Strategy]
    --> [Training: AdamW Optimizer + Label Smoothing + OneCycleLR]
    --> [Best Model Saved (based on Validation Accuracy)]
    --> [Evaluation: Accuracy, Precision, Recall, F1-Score]
```

## Workflow Steps

### 1. Preprocessing & Data Augmentation

Instead of a separate, offline preprocessing step, we employ a powerful *on-the-fly augmentation* strategy using the albumentations library. This is defined directly in the `train_gender.py` script and ensures the model sees a unique, degraded version of an image at every single training step.

- **Library Used:** albumentations for its high performance and rich set of transformations.
- **Techniques Applied to Training Data:**
    - **Geometric:** HorizontalFlip (p=0.5).
    - **Degradation Simulation (OneOf block):** A random choice of MotionBlur, GaussNoise, RandomBrightnessContrast, RandomFog, or RandomRain is applied to simulate challenging environments.
    - **Regularization (CoarseDropout):** Randomly erases parts of the image, forcing the model to learn a more holistic representation of the face.
- **Validation Data:** Only resized and normalized to provide a stable evaluation benchmark.

### 2. Model Architecture

- **Backbone:** **ConvNeXt-Tiny**, pre-trained on the massive **ImageNet-22K** dataset.
- **Rationale:** ConvNeXt is a modern, high-performance Convolutional Neural Network (CNN) that incorporates design principles from Vision Transformers. Using the `.in22k` pre-trained weights gives our model a vastly superior understanding of general visual features compared to standard ImageNet-1K models.
- **Head:** A simple `torch.nn.Linear` layer is attached to the backbone to output predictions for our 2 classes (Male / Female).

### 3. Training Strategy

Our training methodology is designed for maximum performance and stability through a combination of advanced techniques.

- **Two-Stage Fine-Tuning:**
    1. **Stage 1 (Head Training):** Freeze ConvNeXt backbone, train classification head.
    2. **Stage 2 (Full Fine-Tuning):** Unfreeze all layers, fine-tune with low learning rate.
- **Optimizer:** AdamW
- **Loss Function:** Cross-Entropy Loss with Label Smoothing
- **Callbacks & Schedulers:**
    - OneCycleLR
    - EarlyStopping (patience=4)
    - ModelCheckpoint (save best model only)

### 4. Evaluation & Inference

- **Metrics:** Accuracy, Precision, Recall, and F1-Score (via sklearn)
- **Process:** Run `test_gender.py` to evaluate saved best model on validation set.

## Requirements

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Configuration
Edit `config.py` under the "TASK A" section:
```python
# Example
DATASET_FOLDER_NAME = "NewDataset"
GENDER_MODEL_NAME = "convnext_tiny"
GENDER_BATCH_SIZE = 64
```

### 2. Train
```bash
python train_gender.py
```

### 3. Evaluate
```bash
python test_gender.py
```

## Results

Final Evaluation Report:

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.9455  |
| Precision | 0.9678  |
| Recall    | 0.9650  |
| F1-Score  | 0.9664  |

##  Contributors

- **Ishan Peshkar**
