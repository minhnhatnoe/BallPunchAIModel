from typing import List, Tuple
import numpy as np
import torch
from torch import nn


def load_data(paths: 'List[str, str]', *args, **kwargs) -> 'List[np.ndarray, np.ndarray]':
    data = [np.load(path, *args, **kwargs) for path in paths]
    for array in data:
        assert(array.shape[0] == data[0].shape[0])
    return data


def save_model(model: nn.Module, optimizer: torch.optim.Adam, path: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_model(model: nn.Module, optimizer: torch.optim.Adam | None, path: str) -> None:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

