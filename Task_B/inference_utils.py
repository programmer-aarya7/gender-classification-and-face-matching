import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

class SiameseNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.projector = torch.nn.Linear(1280, embedding_dim)

    def forward_once(self, x):
        features = self.backbone(x)
        return self.projector(features)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

def load_model(checkpoint_path):
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def get_embedding(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = val_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    return emb.squeeze(0)

def euclidean_distance(emb1, emb2):
    return torch.norm(emb1 - emb2).item()
