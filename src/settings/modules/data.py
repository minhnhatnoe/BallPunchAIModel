from os.path import join, pardir, realpath


def get_real_path(*args) -> str:
    return str(realpath(join(*args)))


parent_path = get_real_path(__file__, pardir, pardir, pardir, pardir)

data_path = get_real_path(parent_path, "data")
raw_dataset_path = get_real_path(data_path, "RAW")

train_data = [
    get_real_path(data_path, "image.npy"),
    get_real_path(data_path, "label.npy")
]
train_data_full = [
    get_real_path(data_path, "image_full.npy"),
    get_real_path(data_path, "label_full.npy")
]
train_data_part = [
    get_real_path(data_path, "image_part.npy"),
    get_real_path(data_path, "label_part.npy")
]
tests_data = [
    get_real_path(data_path, "tests.npy"),
    get_real_path(data_path, "names.npy")
]

output_path = get_real_path(parent_path, "results")


def get_model_path(model_name: str) -> str:
    return get_real_path(output_path, f"{model_name}_state_dict.pt")


result_path = get_real_path(output_path, "result.csv")

kaggle_path = get_real_path(parent_path, "src", "helper")
