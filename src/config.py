from os import path

x_path = str(path.realpath(path.join(__file__, path.pardir, "image.npy")))
y_path = str(path.realpath(path.join(__file__, path.pardir, "label.npy")))

test_size = 0.2
kfold_nsplits = 20
batch_size = 256
seed = 42
