import torch
from torch import nn
from torchvision import transforms
from settings.modules.dev import device

def get_random_transforms() -> nn.Sequential:
    random_rate = 0.2
    random_transforms = torch.nn.Sequential(
        transforms.RandomAdjustSharpness(random_rate),
        transforms.ColorJitter(brightness=random_rate, contrast=random_rate, saturation=random_rate, hue=random_rate),
        transforms.RandomResizedCrop((224, 224), (0.8, 0.8), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(random_rate),
        transforms.RandomVerticalFlip(random_rate)
    )
    random_transforms = random_transforms.to(device)
    return random_transforms
