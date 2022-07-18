import torch
from torch import nn
from torchvision import transforms
from settings.modules.dev import device

def get_random_transforms() -> nn.Sequential:
    random_rate = 0.5
    random_transforms = torch.nn.Sequential(
        transforms.RandomAdjustSharpness(random_rate),
        transforms.ColorJitter(brightness=random_rate, contrast=random_rate, saturation=random_rate, hue=random_rate),
        transforms.RandomCrop((224, 224), padding=[0], pad_if_needed=True, padding_mode="symmetric"),
        transforms.RandomHorizontalFlip()
    )
    random_transforms = random_transforms.to(device)
    return random_transforms