from os import path
from torch import nn
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