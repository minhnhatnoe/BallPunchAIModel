from os import path
from typing import Tuple
import torch
from torch import nn
import numpy as np
from torchvision.models import vgg16, VGG16_Weights

data_path = path.realpath(path.join(__file__, path.pardir, path.pardir, "data"))

x_path = str(path.realpath(path.join(data_path, "image.npy")))
y_path = str(path.realpath(path.join(data_path, "label.npy")))
t_path = str(path.realpath(path.join(data_path, "tests.npy")))
n_path = str(path.realpath(path.join(data_path, "names.npy")))

model_path = str(path.realpath(path.join(__file__, path.pardir, "model_state_dict.pt")))

import_batch = 10000
test_size = 0.2
kfold_nsplits = 10
kfold_nrepeats = 4
batch_size = 32
seed = 42

def get_model() -> nn.Module:
    model = vgg16(weights=VGG16_Weights.DEFAULT, progress=True)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model

class Data(torch.utils.data.Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, indices: np.ndarray) -> None:
        assert(image.shape[0] == label.shape[0])
        self.image = image
        self.label = label
        self.indices = indices
    
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, idx: int) -> 'Tuple[np.ndarray, bool]':
        idx = self.indices[idx]
        return self.image[idx], self.label[idx]
