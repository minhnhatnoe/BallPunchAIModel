from typing import Tuple
from os import path
import numpy as np
import torch
from torch import nn
from settings.modules import data, utils
from helper import loader, debug, boilerplate

class TrainConfig:
    def __load_model(self, model_settings: Tuple[nn.Module, str]) -> None:
        self.model, self.model_name = model_settings
        if self.use_grayscale:
            self.model_name += '_gs'
        self.optimizer = utils.get_adam(self.model.parameters())

        self.__model_path = data.get_model_path(self.model_name)
        if path.exists(self.__model_path):
            print(f"Resuming from {self.__model_path}")
            loader.load_model(self.model, self.optimizer, self.__model_path)
        else:
            print(f"Creating new model at {self.__model_path}")
        debug.print_model_size(self.model)

    def __load_data(self) -> None:
        self.dataset_image, self.dataset_label = \
            loader.load_data(self.__data_path, mmap_mode='c')

    def __load_transforms(self) -> None:
        self.transforms = utils.get_random_transforms()
        if self.use_grayscale:
            self.grayscale = utils.get_grayscale_transform()

    def __load_loss(self) -> None:
        hit = self.dataset_label.sum()
        self.data_class = (self.dataset_label.shape[0] - hit, hit)
        print(f'''Stats:
        | Number of not-punching: {self.data_class[0]} \t| Number of punching: {self.data_class[1]}''')

        self.criterion = utils.get_cross_entropy_loss(self.data_class)

    def get_split(self):
        split_generator = utils.get_kfold_class(10, 1000)
        return split_generator.split(self.dataset_image)

    def __init__(self,
                 model_settings: Tuple[nn.Module, str],
                 train_paths: str = data.train_data_full,
                 batch_size: int = 32,
                 use_grayscale: bool = False) -> None:
        self.device = utils.device
        self.batch_size = batch_size
        self.use_grayscale = use_grayscale
        self.__data_path = train_paths
        self.__load_model(model_settings)
        self.__load_data()
        self.__load_transforms()
        self.__load_loss()

    def save_checkpoint(self) -> None:
        loader.save_model(self.model, self.optimizer, self.__model_path)

    def get_dataloader(self, indices: np.ndarray) -> None:
        data = boilerplate.TrainingDataset(
            self.dataset_image, self.dataset_label, indices)
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True)

    def load_best(self) -> None:
        loader.load_model(self.model, None, self.__model_path)

device = utils.device
train_paths = data.train_data_full
tests_paths = data.tests_data
result_path = data.result_path

kaggle_path = data.kaggle_path
raw_dataset_path = data.raw_dataset_path

import_batch = 1000
test_size = 0.2

des_sequence_early_stop = 5
early_stop = 10
