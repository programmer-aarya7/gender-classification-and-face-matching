import os
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP')

def is_valid_file(filename):
    return filename.endswith(VALID_EXTENSIONS)

transform = A.Compose([
    A.Resize(112, 112),
    A.RandomBrightnessContrast(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

class AlbumentationsImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, is_valid_file=is_valid_file)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = np.array(Image.open(path).convert("RGB"))
        if self.albumentations_transform:
            image = self.albumentations_transform(image=image)["image"]
        return image, label

DATA_PATH = "Task_B/train_yolo_aligned"
print("🔍 Listing folders inside:", DATA_PATH)
for f in os.listdir(DATA_PATH):
    print("📁", f)

dataset = AlbumentationsImageFolder(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

print("✅ Dataloader ready.")
print(" Total classes:", len(dataset.classes))
print(" Total images :", len(dataset))
