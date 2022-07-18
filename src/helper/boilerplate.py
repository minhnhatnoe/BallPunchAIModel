from typing import Tuple
import torch
import numpy as np

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, indices: np.ndarray) -> None:
        assert(image.shape[0] == label.shape[0])
        self.image = image
        self.label = label
        self.indices = indices
    
    def __len__(self) -> int:
        return self.indices.shape[0]
    
    def __getitem__(self, idx: int) -> 'Tuple[np.ndarray, bool]':
        idx = self.indices[idx]
        return self.image[idx], self.label[idx]
