import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision import models

_model = None
_preprocess = None

def get_resnet_model():
    global _model
    if _model is None:
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        base_model.eval()
        _model = torch.nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
    return _model

def get_preprocess():
    global _preprocess
    if _preprocess is None:
        _preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return _preprocess

def get_resnet_embedding_model():
    global _model
    if _model is None:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.eval()
        _model = torch.nn.Sequential(*list(base.children())[:-1])  # remove classifier
    return _model