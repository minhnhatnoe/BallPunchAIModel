from torch import nn
from settings.modules.dev import device

def get_cross_entropy_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss()
    return criterion.to(device)
