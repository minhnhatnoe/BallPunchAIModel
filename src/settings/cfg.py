import random
from sklearn.model_selection import RepeatedKFold
from settings.modules import models, trans, loss, dev, data, optim, kf

random.seed(42)

device = dev.device
model, model_name = models.get_vgg16(True)
transforms = trans.get_random_transforms()
get_loss = loss.get_cross_entropy_loss
optimizer = optim.get_adam(model.parameters())

def get_idx_gen() -> RepeatedKFold:
    return kf.get_kfold_class(split_count=10, loop_count=10, seed=random.randint(0, 1000))

train_paths = data.train_data_full
tests_paths = data.tests_data

model_path = data.get_model_path(model_name)
result_path = data.result_path

kaggle_path = data.kaggle_path
raw_dataset_path = data.raw_dataset_path

import_batch = 1000
test_size = 0.2
batch_size = 32

early_stop = 4
