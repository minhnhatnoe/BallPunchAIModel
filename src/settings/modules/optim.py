import torch


def get_adam(*args) -> torch.optim.Adam:
    return torch.optim.Adam(*args, lr=0.000001)
