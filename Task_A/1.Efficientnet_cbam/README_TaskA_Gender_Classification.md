
#  Gender Classification under Degraded Visual Conditions - Task A

This repository contains the implementation for the **Gender Classification** task where the goal is to classify gender (Male/Female) from facial images even under **challenging visual conditions** such as blur, fog, overexposure, and lighting distortion.

##  Objective

- Predict gender from face images that may be:
  - Motion-blurred
  - Overexposed or foggy
  - Poor lighting / occlusion
  - Partially occluded

##  Pipeline Overview

A[Raw Images (Male/Female) under Visual Distortions] --> B[Preprocessing: DeepLabV3+ + YOLOv8n]
B --> C[Albumentations: Data Augmentation]
C --> D[Balanced Dataset Handling (In-Code Sampling)]
D --> E[EfficientNetB4 + CBAM Backbone]
E --> F[Training with Label Smoothing + Class Weights]
F --> G[ModelCheckpoint: Best Model Saved]
G --> H[Evaluation: Accuracy, Precision, Recall, F1-Score]

##  Why Preprocessed Data is Not Included

To keep the repository lightweight and within GitHub's file size limits, **raw and preprocessed datasets are not uploaded**.
However, you can **recreate them easily using the scripts** provided below.


## 📁 Expected Folder Structure

```bash
1.Efficientnet_cbam/
├── train/                         # Raw training data
│   ├── male/
│   └── female/
│
├── val/                           # Raw validation data
│   ├── male/
│   └── female/
│
├── deeplab_yolo_preprocessed/    # Output from `deeplab_yolo_preprocess.py`
│   ├── male/
│   └── female/
│
├── augmented/                    # Output from `augment_with_albumentations.py`
│   ├── male/
│   └── female/
│
├── yolo_val_preprocessed/        # Output from `val_yolo_preprocess.py`
│   ├── male/
│   └── female/
│
├── augmented_val/                # Output from `val_augment_withalbumentation.py`
│   ├── male/
│   └── female/
│
├── model/                        # Saved model checkpoints
│
├── test_predictions/             # Inference results
│
├── confusion_matrix.png          # Evaluation heatmap
│
├── yolov8n.pt                    # YOLOv8n weights
│
├── deeplabv3_model.py
├── deeplab_yolo_preprocess.py
├── augment_with_albumentations.py
├── val_yolo_preprocess.py
├── val_augment_withalbumentation.py
├── train_efficientnet_cbam.py
├── Testing.py
├── requirements.txt
├── README_TaskA_Gender_Classification.md
├── gender.txt                    # Classification report (optional)
```

##  Step-by-Step Workflow

### 1. Preprocessing (Training Data)

```bash
python deeplab_yolo_preprocess.py
```

- Segments faces using DeepLabV3+
- Detects faces using YOLOv8n
- Saves clean 224x224 face crops

### 2. Preprocessing (Validation Data)

```bash
python val_yolo_preprocess.py
```

- Same as above but for validation images

### 3. Data Augmentation

```bash
python augment_with_albumentations.py
python val_augment_withalbumentation.py
```

- Applies:
  - Horizontal Flip
  - Brightness/Contrast
  - Gaussian Noise
  - Rotation & Blur

### 4. Model Architecture

- **Backbone**: EfficientNet-B4
- **Attention**: CBAM (Convolutional Block Attention Module)
- **Head**: 2-class softmax for Male/Female

### 5. Training

```bash
python train_efficientnet_cbam.py
```

- Uses:
  - Label smoothing
  - Class weights (handled by `WeightedRandomSampler`)
  - ModelCheckpointing (`best_model.pth`)
  - ReduceLROnPlateau

### 6. Evaluation

```bash
python Testing.py
```

- Generates:
  - Classification report
  - Confusion matrix
  - 20 inference visualizations with predicted vs. actual class

##  Install Dependencies

```bash
pip install -r requirements.txt
```

- Make sure `yolov8n.pt` is present in the root folder.

##  Results

              precision    recall  f1-score   support

      female       0.77      0.93      0.84       243
        male       0.98      0.92      0.95       845

    accuracy                           0.92      1088
   macro avg       0.87      0.93      0.90      1088
weighted avg       0.93      0.92      0.92      1088

##  Contributor

- **Aarya Shetiye**

