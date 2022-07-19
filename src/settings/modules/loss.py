from typing import List
from torch import nn, Tensor
from settings.modules.dev import device


def get_cross_entropy_loss(cls_cnt: 'List[int]') -> nn.CrossEntropyLoss:
    total = sum(cls_cnt)
    class_weights = [1 - cnt/total for cnt in cls_cnt]
    criterion = nn.CrossEntropyLoss(weight=Tensor(class_weights))
    return criterion.to(device)
