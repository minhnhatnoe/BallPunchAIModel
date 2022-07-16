import os
from os import path
from glob import glob
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms
from npy_append_array import NpyAppendArray as BigArray
import config

dataset_path = str(path.realpath(path.join(config.data_path, "RAW")))
video_file_names = ["VID1", "VID3", "VID4", "VID5", "VID6"]
image_folder_names = ["video_10", "video_11", "video_20", "video_40",
                      "video_41", "video_60", "video_110", "video_120", "video_130"]
test_folder_names = ["video_12", "video_20a", "video_20b", "video_40a", "video_42", 
                      "video_60a", "video_111", "video_120a", "video_131"]

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")

'''Transformation list'''
transform = transforms.Sequential([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomHorizontalFlip()
])

def transform_array(image_batch: torch.FloatTensor) -> torch.FloatTensor:
    '''Transform an array of image'''
    image_batch = image_batch.to(device)
    image_batch /= 255.0
    image_batch = transform(image_batch)
    image_batch = image_batch.cpu().detach()
    return image_batch


def split_append_array(total_image: BigArray, image: np.ndarray) -> None:
    per_batch = config.import_batch
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


def load_video(total_image: BigArray, video_name: str) -> None:

    image_path = path.join(dataset_path, "image", f"{video_name}_Extract.npy")

    print(f"Loading {video_name}")
    image_array = np.load(image_path)
    assert((image_array.shape[1], image_array.shape[2]) == (224, 224))

    print(f"Processing {video_name}")
    split_append_array(total_image, image_array)


def load_video_label(total_label: BigArray, video_name: str) -> None:
    label_path = path.join(dataset_path, "label", f"{video_name}.npy")

    print(f"Loading label {video_name}")
    label_array = np.load(label_path)
    total_label.append(label_array)

def image_folder_append(total_image: BigArray, image_paths):
    image_array = []
    batch_num = 1
    for image in image_paths:
        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224))
        image_array.append(img)
        if len(image_array) == config.batch_size:
            print(f"Processing batch {batch_num} of {folder_name}: ", end="")
            batch_num += 1
            image_array = np.array(image_array)
            split_append_array(total_image, image_array)
            image_array = []
            print("Finished.")

    image_array = np.array(image_array)
    split_append_array(total_image, image_array)


def load_image_folder(total_image: BigArray, folder_name: str) -> None:
    print(f"Loading {folder_name}")
    image_paths = glob(path.join(dataset_path, "image", folder_name, "*"))
    image_paths = sorted(image_paths)
    image_folder_append(total_image, image_paths)

def load_image_label(total_label: BigArray, folder_name: str) -> None:
    label_path = path.join(dataset_path, "label", f"{folder_name}.csv")
    label_array = []
    print(f"Loading label {folder_name}")
    with open(label_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            label_array.append(row[1] == '0') # The dataset is inverted
    label_array = np.array(label_array)
    total_label.append(label_array)

def load_image_tests(total_image: BigArray, folder_name: str) -> None:
    print(f"Loading {folder_name}")
    image_paths = glob(path.join(dataset_path, "test", folder_name, "*"))
    image_paths = sorted(image_paths)
    image_folder_append(total_image, image_paths)

def create_file_array(total_filename: BigArray, folder_name: str) -> None:
    print(f"Loading names of {folder_name}")
    image_paths = glob(path.join(dataset_path, "test", folder_name, "*"))
    image_paths = sorted(image_paths)
    image_paths = [str(path.basename(file_path)) for file_path in image_paths]
    total_filename.append(np.array(image_paths))

def load_punch(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    add = 0
    for img, lbl in zip(image, label):
        if lbl == 1:
            add += 1
    print(add)
if __name__ == '__main__':
    if os.path.exists(config.x_path):
        os.remove(config.x_path)
    if os.path.exists(config.y_path):
        os.remove(config.y_path)
    if os.path.exists(config.t_path):
        os.remove(config.t_path)
    if os.path.exists(config.n_path):
        os.remove(config.n_path)

    with BigArray(config.x_path) as total_image:
        with BigArray(config.y_path) as total_label:
            for video_name in video_file_names:
                load_video(total_image, video_name)
                load_video_label(total_label, video_name)
            for folder_name in image_folder_names:
                load_image_folder(total_image, folder_name)
                load_image_label(total_label, folder_name)
    punch_image = load_punch(np.load(config.x_path), np.load(config.y_path))
    with BigArray(config.t_path) as total_image:
        with BigArray(config.n_path) as total_filename:
            for folder_name in test_folder_names:
                load_image_tests(total_image, folder_name)
                create_file_array(total_filename, folder_name)

