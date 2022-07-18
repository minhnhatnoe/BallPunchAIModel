import torch
from torch import nn

def print_model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    param_size  = param_size / 1024**2
    buffer_size = buffer_size / 1024**2
    print(f'Param size: {param_size:.3f}MB')
    print(f'Buffer size: {buffer_size:.3f}MB')

def print_tensor_size(tensor: torch.Tensor):
    size_gb = tensor.element_size() * tensor.nelement() / (1<<30)
    print(f"{size_gb:.3f}GB")