from os import path
import shutil
from torch import nn
import torch
from torchvision import transforms
import numpy as np
import os
import models

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")


src_path = path.join(__file__, path.pardir)
data_path = path.join(src_path, path.pardir, "data")

x_path = str(path.realpath(path.join(data_path, "image.npy")))
y_path = str(path.realpath(path.join(data_path, "label.npy")))
t_path = str(path.realpath(path.join(data_path, "tests.npy")))
n_path = str(path.realpath(path.join(data_path, "names.npy")))

model_path = str(path.realpath(path.join(src_path, "model_state_dict.pt")))
result_path = str(path.realpath(path.join(src_path, "result.csv")))


import_batch = 10000
test_size = 0.2
kfold_nsplits = 10
kfold_nrepeats = 2
batch_size = 64
seed = 42
random_rate = 0.5

def get_model() -> nn.Module:
    # model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT, progress=True)
    return models.get_vgg11(pretrained=True)
    return model

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv = [
#             nn.Conv2d(3, 64, 3, padding = 1), 
#             nn.Conv2d(64, 128, 3, padding = 1), 
#             nn.Conv2d(128, 256, 3, padding = 1),
#             nn.Conv2d(256, 512, 3, padding = 1)
#         ]
#         self.conv = nn.ModuleList(self.conv)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = [
#             nn.Linear(7 * 7 * 512, 4096),
#             nn.Linear(4096, 4096),
#             nn.Linear(4096, 1000)
#         ]
#         self.fc = nn.ModuleList(self.fc)
#         self.final = nn.Linear(1000, 2)

#         self.labels_count = 2

#     def forward(self, inputs):
#         x = self.pool(F.relu(self.conv[0](inputs)))

#         for i in range(1, len(self.conv)):
#             x = self.pool(F.relu(self.conv[i](x)))

#         x = self.pool(x)
#         # print(x.shape)

#         x = x.view(-1, 7 * 7 * 512)
#         # print(x.shape)

#         for i in self.fc:
#             x = F.relu(i(x))

#         x = self.dropout(x)
#         x = self.final(x)

#         preds = nn.LogSoftmax(dim = 1)(x)
#         return preds

# def get_model2() -> nn.Module:
#     return Net()

torch.manual_seed(seed)
# torch.use_deterministic_algorithms(mode=True)

random_transforms = torch.nn.Sequential(
    transforms.RandomAdjustSharpness(random_rate),
    transforms.ColorJitter(brightness=random_rate, contrast=random_rate, saturation=random_rate, hue=random_rate),
    transforms.RandomCrop((224, 224), padding=[0], pad_if_needed=True, padding_mode="symmetric"),
    transforms.RandomHorizontalFlip()
)

random_transforms = random_transforms.to(device)

def load_memmap_rplus(original_path: str) -> np.ndarray:
    temp_path = path.splitext(original_path)
    temp_path = temp_path[0] + "_temp" + temp_path[1]
    shutil.copy(original_path, temp_path)
    return np.load(temp_path, mmap_mode='r+')
