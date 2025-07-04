#  Face Matching Under Visual Distortions - **Task B**

This repository contains the full implementation for **Face Matching**, where the goal is to **match distorted facial images** to their correct identities using a **Siamese Network** trained with **Contrastive Loss**.

##  Objective

- Build a robust **face verification system** that can:
  - Match distorted or altered face images to clean reference identities.
  - Handle **blur**, **lighting issues**, **pose variations**, and **other degradations**.

- Use a **Siamese neural network** that learns to project images of the same identity close together in the embedding space.

##  Pipeline Overview

A[Raw Reference + Distorted Images] --> B[YOLOv8: Face Detection and Cropping]
B --> C[Albumentations: Data Augmentation]
C --> D[PairDataset: Positive & Negative Pairs]
D --> E[Siamese Network with EfficientNetB0]
E --> F[Training: Contrastive Loss]
F --> G[Best Model Saved]
G --> H[Inference: Match distorted image to reference]

##  Workflow Breakdown

### 1. Preprocessing

- **YOLOv8n** detects and crops the face from reference and distorted images.
- Faces are resized to **112x112**.
- Output is saved in:

```bash
Task_B/train_yolo_aligned/
Task_B/val_yolo_aligned/
```

Scripts:
```bash
python face_preprocessing.py
python face_valpreprocessing.py
```

### 2. Data Augmentation

- Use `Albumentations` to simulate real-world distortions:
  - Brightness/Contrast
  - Gaussian Noise
  - Blur, Flip, Rotation
- Output used for **PairDataset** training.

Script:
```bash
python albumentations_dataloader_imagefolder.py
```

### 3. Pair Generation

- Build **positive pairs**: (reference vs distorted of same person)
- Build **negative pairs**: (reference vs distorted of different persons)
- Pairs are used for Siamese training.

### 4. Siamese Network Architecture

- Backbone: **EfficientNetB0** without classification head
- Output: **512-dimensional embeddings**
- Trained using **Contrastive Loss**:

```python
ContrastiveLoss = (1 - label) * D^2 + label * max(margin - D, 0)^2
```

- Where:
  - `D`: Euclidean distance between embeddings
  - `label`: 0 for same, 1 for different
  - `margin`: Distance threshold (e.g., 1.0)

### 5. Training

Train using:
```bash
python face_recog_train_siamese.py
```

Saves best model to:
```bash
checkpoints/best_model.pth
```

### 6. Inference & Evaluation

Evaluate using:
```bash
python face_evaluate.py
```

Outputs:
- Matching result images: `sample_predictions/`
- Evaluation scores: `evaluation_metrics.txt`
- Confusion matrix plot: `sample_predictions/confusion_matrix.png`

## 📁 Expected Folder Structure

```bash
Task_B/
├── train/                             # Raw training images
│   ├── identity_1/
│   └── identity_1/distortion/
│
├── val/                               # Raw validation images
│   ├── identity_1/
│   └── identity_1/distortion/
│
├── preprocessed_images/              # (Optional)
├── checkpoints/                      # Trained model (best_model.pth)
├── sample_predictions/               # Saved prediction result images
├── val_preprocessing_checkpoint.json # Checkpoint tracker
├── yolov8n.pt                        # YOLOv8 weights
│
├── albumentations_dataloader_imagefolder.py
├── face_preprocessing.py
├── face_valpreprocessing.py
├── face_recog_train_siamese.py
├── face_evaluate.py
├── inference_utils.py
│
├── evaluation_metrics.txt            # Final performance metrics
├── requirements.txt
├── README_TaskB_Face_Matching.md     # This README file
```

##  Evaluation Metrics

Stored in `evaluation_metrics.txt`

Evaluation Metrics:
**Accuracy:  0.9716**
**Precision: 1.0000**
**Recall:    0.9716**
**F1 Score:  0.9856**

##  How to Run

### Install Requirements

```bash
pip install -r requirements.txt
```

Ensure `yolov8n.pt` is present in the root directory.

### Preprocess Data (YOLO + Resize)

```bash
python face_preprocessing.py
python face_valpreprocessing.py
```

### Augment Faces (Optional)

```bash
python albumentations_dataloader_imagefolder.py
```

### Train Siamese Network

```bash
python face_recog_train_siamese.py
```

### Evaluate & Save Predictions

```bash
python face_evaluate.py
```
## Contributors

- **Aarya Shetiye**