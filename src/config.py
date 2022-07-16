from os import path

data_path = path.realpath(path.join(__file__, path.pardir, path.pardir, "data"))

x_path = str(path.realpath(path.join(data_path, "image.npy")))
y_path = str(path.realpath(path.join(data_path, "label.npy")))
t_path = str(path.realpath(path.join(data_path, "tests.npy")))

import_batch = 10000
test_size = 0.2
kfold_nsplits = 10
kfold_nrepeats = 4
batch_size = 32
seed = 42
