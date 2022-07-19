from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
__all__ = ['TrainingDataset', 'ExportDataset']

class TrainingDataset(Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, indices: np.ndarray) -> None:
        assert(image.shape[0] == label.shape[0])
        self.image = image
        self.label = label
        self.indices = indices.copy()
        np.random.shuffle(self.indices)
        
    
    def __len__(self) -> int:
        return self.indices.shape[0]
    
    def __getitem__(self, idx: int) -> 'Tuple[np.ndarray, bool]':
        idx = self.indices[idx]
        return self.image[idx], self.label[idx]

class ExportDataset(Dataset):
    def __init__(self, image: np.ndarray, name: np.ndarray) -> None:
        assert(image.shape[0] == name.shape[0])
        self.image = image
        self.name = name
    
    def __len__(self) -> int:
        return self.image.shape[0]
    
    def __getitem__(self, idx: int) -> 'Tuple[np.ndarray, str]':
        return self.image[idx], self.name[idx]
