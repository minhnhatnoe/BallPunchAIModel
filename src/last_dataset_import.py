import os
from os import path
from glob import glob
from typing import Tuple, List
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms
from npy_append_array import NpyAppendArray as BigArray
from settings import cfg
from helper import loader

def transform_array(image_batch: torch.FloatTensor) -> torch.FloatTensor:
    '''Transform an array of image'''
    image_batch = image_batch.to(cfg.device, dtype=torch.float32)
    image_batch /= 255.0
    image_batch = image_batch.cpu().detach()
    return image_batch


def split_append_array(total_image: BigArray, image: np.ndarray) -> None:
    per_batch = cfg.import_batch
    image_count = image.shape[0]
    for i in range(0, image_count, per_batch):
        last_idx = min(i + per_batch, image.shape[0])
        image_batch = image[i:last_idx]

        image_batch = image_batch.astype(float)
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        image_batch = torch.from_numpy(image_batch)
        image_batch = transform_array(image_batch).numpy()

        image_batch = np.ascontiguousarray(image_batch)
        total_image.append(image_batch)


def image_folder_append(total_image: BigArray, image_paths: 'List[str]') -> int:
    image_array = []
    total_added = 0
    batch_num = 1
    for image in image_paths:
        img = cv2.imread(image)
        try:
            assert(np.isnan(img).sum()==0)
            assert(img.shape[0] * img.shape[1] * img.shape[2] != 0)
        except:
            print(img, type(img))

        img = cv2.resize(img, (224, 224))
        image_array.append(img)
        if len(image_array) == cfg.import_batch:
            print(f"Processing batch {batch_num}: ", end="")
            batch_num += 1
            total_added += len(image_array)
            image_array = np.array(image_array)
            split_append_array(total_image, image_array)
            image_array = []
            print("Finished.")

    print(
        f"Processing last batch ({len(image_array)} images): ", end="")
    total_added += len(image_array)
    image_array = np.array(image_array)
    split_append_array(total_image, image_array)
    print("Finished.")
    return total_added


def load_image_tests(total_image: BigArray, folder_name: str) -> int:
    print(f"Loading {folder_name}")
    image_paths = glob(path.join(folder_name, "*"))
    image_paths = sorted(image_paths)
    return image_folder_append(total_image, image_paths)


def get_file_name(file_path: str) -> str:
    file_path = path.basename(file_path)
    return str(file_path)


def create_file_array(total_filename: BigArray, folder_name: str) -> int:
    print(f"Loading names of {folder_name}")
    image_paths = glob(path.join(folder_name, "*"))
    image_paths = sorted(image_paths)
    image_paths = [get_file_name(file_path)
                   for file_path in image_paths]
    image_paths = np.array(image_paths, dtype='=U32')
    total_filename.append(image_paths)
    return len(image_paths)


if __name__ == '__main__':
    if os.path.exists(cfg.tests_paths[0]):
        os.remove(cfg.tests_paths[0])
    if os.path.exists(cfg.tests_paths[1]):
        os.remove(cfg.tests_paths[1])

    with BigArray(cfg.tests_paths[0]) as total_image:
        with BigArray(cfg.tests_paths[1]) as total_filename:
            x = load_image_tests(total_image, "/home/phuonghd/NHAT/BallPunchAIModel/data/RAW/Data")
            y = create_file_array(total_filename, '/home/phuonghd/NHAT/BallPunchAIModel/data/RAW/Data')
            assert(x == y)
