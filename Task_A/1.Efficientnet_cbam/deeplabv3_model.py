import torch
import torchvision.models.segmentation as models

def load_deeplab():
    model = models.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model
