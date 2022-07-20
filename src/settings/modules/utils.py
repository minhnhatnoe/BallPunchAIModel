from typing import List
import torch
from torch import nn
from torchvision import transforms
from sklearn.model_selection import RepeatedKFold

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")


def get_kfold_class(split_count: int, loop_count: int) -> RepeatedKFold:
    return RepeatedKFold(n_splits=split_count, n_repeats=loop_count)


def get_random_transforms() -> nn.Sequential:
    random_transforms = torch.nn.Sequential(
        transforms.RandomAdjustSharpness(0.2),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomResizedCrop((224, 224), (0.8, 0.8), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.2)
    )
    return random_transforms.to(device)


def get_adam(*args, **kwargs) -> torch.optim.Adam:
    return torch.optim.Adam(*args, **kwargs)

def get_cross_entropy_loss(cls_cnt: 'List[int]') -> nn.CrossEntropyLoss:
    total = sum(cls_cnt)
    class_weights = [1 - cnt/total for cnt in cls_cnt]
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
    return criterion.to(device)
