from os import path

data_path = path.realpath(path.join(__file__, path.pardir, path.pardir, "data"))

x_path = str(path.realpath(path.join(data_path, "image.npy")))
y_path = str(path.realpath(path.join(data_path, "label.npy")))

test_size = 0.2
kfold_nsplits = 2
batch_size = 256
seed = 42
