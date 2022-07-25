import cv2
import numpy as np
import os
from os import path
import csv
from settings import cfg
dataset_path = cfg.raw_dataset_path

video_file_names = ["VID1", "VID3", "VID4", "VID5", "VID6"]
def write_file(images: np.ndarray, label: np.ndarray, img_path: str, label_path: str) -> None:
    cnt = 0
    with open(label_path, "w", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Label"])
        for img, lbl in zip(images, label):
            img_name = f"frame_{cnt:06d}.png"
            if cnt == 0:
                print(str(os.path.join(img_path, img_name)))
            cv2.imwrite(str(os.path.join(img_path, img_name)), img)
            writer.writerow([img_name, str(int(lbl))])
            cnt += 1



for name in video_file_names:
    image_path = str(path.realpath(path.join(dataset_path, "image", f"{name}_Extract.npy")))
    label_path = str(path.realpath(path.join(dataset_path, "label", f"{name}.npy")))
    images = np.load(image_path, mmap_mode = "r")
    label = np.load(label_path, mmap_mode = "r")

    image_folder_write = str(path.realpath(path.join(dataset_path, "export", name)))
    label_write_path = str(path.realpath(path.join(dataset_path, "label_export", f"{name}.csv")))
    print(image_folder_write)
    write_file(images, label, image_folder_write, label_write_path)
    print(f"{name} done")
