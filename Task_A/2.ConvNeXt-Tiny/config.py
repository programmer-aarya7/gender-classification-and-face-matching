import torch
import os

# ===================================================================
# --- GLOBAL PROJECT & DATASET CONFIGURATION ---
# ===================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "NewDataset" # Or "Dataset"
BASE_DIR = os.path.join(PROJECT_ROOT, DATASET_FOLDER_NAME)

# ===================================================================
# --- GLOBAL DEVICE CONFIGURATION ---
# ===================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
NUM_WORKERS = 4

# ===================================================================
# --- TASK A: GENDER CLASSIFICATION CONFIGURATIONS ---
# ===================================================================
# (We will keep using ConvNeXt for gender classification as it works well)
GENDER_MODEL_NAME = "convnext_tiny.in22k"
GENDER_TRAIN_DIR = os.path.join(BASE_DIR, "Task_A", "train")
GENDER_VAL_DIR = os.path.join(BASE_DIR, "Task_A", "val")
NUM_CLASSES_GENDER = 2
CLASS_LABELS_GENDER = ['female', 'male']
GENDER_BATCH_SIZE = 64
GENDER_WEIGHT_DECAY = 0.01
GENDER_LABEL_SMOOTHING = 0.1
GENDER_FT_HEAD_EPOCHS = 5
GENDER_LR_HEAD = 1e-3
GENDER_FT_FULL_EPOCHS = 15
GENDER_LR_FULL = 1e-5
GENDER_EARLY_STOPPING_PATIENCE = 4
GENDER_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, f"gender_classifier_{GENDER_MODEL_NAME}_{DATASET_FOLDER_NAME}.pth.tar")

