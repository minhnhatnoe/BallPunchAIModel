import torch

def get_adam() -> torch.optim.Adam:
    return torch.optim.Adam()