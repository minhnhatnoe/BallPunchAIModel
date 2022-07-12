import os
from os import path
from typing import Tuple
import numpy as np
import torch
from torchvision import transforms
from npy_append_array import NpyAppendArray
import config
dataset_path = str(path.realpath(path.join(config.data_path, "RAW")))

filenames = ["VID1", "VID3", "VID4", "VID5", "VID6"]

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tensor_transform = transforms.ToTensor()

def transform_array(img_batch: torch.FloatTensor) -> torch.FloatTensor:
    img_batch = img_batch.to(device)
    img_batch /= 255.0
    img_batch = transform(img_batch)
    img_batch = img_batch.cpu().detach()
    return img_batch


def split_append_array(total_img: NpyAppendArray, img: np.ndarray) -> None:
    per_batch = 1000
    image_count = img.shape[0]
    for i in range(0, image_count, per_batch):
        last_idx = min(i + 1000, img.shape[0])
        img_batch = img[i:last_idx]

        img_batch = img_batch.astype(float)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))
        img_batch = torch.from_numpy(img_batch)
        img_batch = transform_array(img_batch).numpy()
        
        img_batch = np.ascontiguousarray(img_batch)
        total_img.append(img_batch)


def get_path(filename: str) -> Tuple(str, str):
    return (
        path.join(dataset_path, f"{filename}_Extract.npy"),
        path.join(dataset_path, f"{filename}.npy")
    )

if __name__ == '__main__':
    if os.path.exists(config.x_path):
        os.remove(config.x_path)
    if os.path.exists(config.y_path):
        os.remove(config.y_path)

    with NpyAppendArray(config.x_path) as array_x:
        with NpyAppendArray(config.y_path) as array_y:
            for name in filenames:
                paths = get_path(name)
                print(f"Loading {paths[0]}")
                image_array = np.load(paths[0])

                print(f"Processing {paths[0]}")
                split_append_array(array_x, image_array[0:10])

                print(f"Loading {paths[1]}")
                current_y = np.load(paths[1])
                array_y.append(current_y[0:10])
