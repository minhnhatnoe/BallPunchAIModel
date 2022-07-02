from os import path
dataset_path = str(path.realpath(
    path.join(
        path.realpath(__file__),
        path.pardir, path.pardir,
        "BallPunchAI", "Dataset", "data"
    )
))
file_names = ["VID1"]
test_size = 0.2
kfold_nsplits = 100
batch_size = 256