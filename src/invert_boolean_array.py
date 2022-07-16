from os import path
import numpy as np
import config

dataset_path = str(path.realpath(path.join(config.data_path, "RAW")))
video_file_names = ["VID1", "VID3", "VID4", "VID5", "VID6"]
for video_name in video_file_names:
    label_path = path.join(dataset_path, "label", f"{video_name}.npy")
    labels = np.load(label_path)
    labels = np.invert(labels)
    np.save(label_path, labels)