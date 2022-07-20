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

dataset_path = cfg.raw_dataset_path
# video_file_names = ["VID1", "VID3", "VID4", "VID5", "VID6"]
video_file_names = ["VID3", "VID4", "VID5", "VID6"]
image_folder_names = ["video_10", "video_11", "video_20", "video_40",
                      "video_41", "video_60", "video_110", "video_120", "video_130"]
test_folder_names = ["video_12", "video_20a", "video_20b", "video_40a", "video_42",
                     "video_60a", "video_111", "video_120a", "video_131"]

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


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


def load_video(total_image: BigArray, video_name: str) -> int:
    video_path = path.join(dataset_path, "image", f"{video_name}_Extract.npy")
    print(f"Loading {video_name}")
    image_array = np.load(video_path)
    assert((image_array.shape[1], image_array.shape[2],
           image_array.shape[3]) == (224, 224, 3))

    print(f"Processing {video_name}")
    split_append_array(total_image, image_array)
    return image_array.shape[0]


def load_video_label(total_label: BigArray, video_name: str) -> int:
    label_path = path.join(dataset_path, "label", f"{video_name}.npy")

    print(f"Loading label {video_name}")
    label_array = np.load(label_path)
    # Heuristic to ensure flipped
    assert(label_array.sum() * 3 < label_array.shape[0])
    total_label.append(label_array)
    return label_array.shape[0]


def image_folder_append(total_image: BigArray, image_paths: 'List[str]') -> int:
    image_array = []
    total_added = 0
    batch_num = 1
    for image in image_paths:
        img = cv2.imread(image)
        if np.isnan(img).sum() or img.shape[0] * img.shape[1] * img.shape[2] == 0:
            print(image)
        img = cv2.resize(img, (224, 224))
        image_array.append(img)
        if len(image_array) == cfg.import_batch:
            print(f"Processing batch {batch_num} of {folder_name}: ", end="")
            batch_num += 1
            total_added += len(image_array)
            image_array = np.array(image_array)
            split_append_array(total_image, image_array)
            image_array = []
            print("Finished.")

    print(
        f"Processing last batch of {folder_name} ({len(image_array)} images): ", end="")
    total_added += len(image_array)
    image_array = np.array(image_array)
    split_append_array(total_image, image_array)
    print("Finished.")
    return total_added


def load_image_folder(total_image: BigArray, folder_name: str) -> int:
    print(f"Loading {folder_name}")
    image_paths = glob(path.join(dataset_path, "image", folder_name, "*"))
    image_paths = sorted(image_paths)
    return image_folder_append(total_image, image_paths)


def load_image_label(total_label: BigArray, folder_name: str) -> int:
    label_path = path.join(dataset_path, "label", f"{folder_name}.csv")
    label_array = []
    print(f"Loading label {folder_name}")
    with open(label_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        first = True
        for row in reader:
            if first:
                # Skips first line
                first = False
                continue
            label_array.append(row[1] == '1')
    label_array = np.array(label_array)
    total_label.append(label_array)
    return label_array.shape[0]


def load_image_tests(total_image: BigArray, folder_name: str) -> int:
    print(f"Loading {folder_name}")
    image_paths = glob(path.join(dataset_path, "test", folder_name, "*"))
    image_paths = sorted(image_paths)
    return image_folder_append(total_image, image_paths)


def get_file_name(file_path: str, folder_name: str) -> str:
    file_path = path.basename(file_path)
    file_path = path.splitext(file_path)[0]
    file_path = f"{folder_name}_{file_path}"
    return str(file_path)


def create_file_array(total_filename: BigArray, folder_name: str) -> int:
    print(f"Loading names of {folder_name}")
    image_paths = glob(path.join(dataset_path, "test", folder_name, "*"))
    image_paths = sorted(image_paths)
    image_paths = [get_file_name(file_path, folder_name)
                   for file_path in image_paths]
    image_paths = np.array(image_paths, dtype='=U32')
    total_filename.append(image_paths)
    return len(image_paths)


def load_punch(total_image: np.ndarray, total_label: np.ndarray) -> 'Tuple[np.ndarray, np.ndarray]':
    result_images = []
    for img, lbl in zip(total_image, total_label):
        if lbl:
            result_images.append(img)
    return np.array(result_images), np.full(len(result_images), 1, dtype=bool)


if __name__ == '__main__':
    if os.path.exists(cfg.train_paths[0]):
        os.remove(cfg.train_paths[0])
    if os.path.exists(cfg.train_paths[1]):
        os.remove(cfg.train_paths[1])
    if os.path.exists(cfg.tests_paths[0]):
        os.remove(cfg.tests_paths[0])
    if os.path.exists(cfg.tests_paths[1]):
        os.remove(cfg.tests_paths[1])

    with BigArray(cfg.train_paths[0]) as total_image:
        with BigArray(cfg.train_paths[1]) as total_label:
            for video_name in video_file_names:
                x = load_video(total_image, video_name)
                y = load_video_label(total_label, video_name)
                assert(x == y)
            for folder_name in image_folder_names:
                x = load_image_folder(total_image, folder_name)
                y = load_image_label(total_label, folder_name)
                assert(x == y)

    punch_image, punch_label = load_punch(*loader.load_data(cfg.train_paths))
    assert(punch_image.shape[0] == punch_label.shape[0])

    total_count = np.load(cfg.train_paths[0], mmap_mode='r').shape[0]
    punch_count = punch_image.shape[0]

    # print(
    #     f"Adding more punches, since there are {punch_count} punch images/{total_count} images")
    # with BigArray(cfg.train_paths[0]) as total_image:
    #     with BigArray(cfg.train_paths[1]) as total_label:
    #         while total_count > punch_count * 2:
    #             added = 0
    #             if punch_image.shape[0] > total_count - 2 * punch_count:
    #                 added = total_count - 2*punch_count
    #                 total_image.append(punch_image[:added])
    #                 total_label.append(punch_label[:added])
    #             else:
    #                 added = punch_image.shape[0]
    #                 total_image.append(punch_image)
    #                 total_label.append(punch_label)

    #             total_count += added
    #             punch_count += added
    #             print(f"{punch_count} punch images/{total_count} images")
    #             print(f"Rate: {punch_count/total_count}")

    with BigArray(cfg.tests_paths[0]) as total_image:
        with BigArray(cfg.tests_paths[1]) as total_filename:
            for folder_name in test_folder_names:
                x = load_image_tests(total_image, folder_name)
                y = create_file_array(total_filename, folder_name)
                assert(x == y)
