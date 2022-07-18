import torch

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")
