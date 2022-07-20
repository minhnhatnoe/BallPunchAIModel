from os import environ
import torch


environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICE"] = "0,1"
torch.manual_seed(42)
