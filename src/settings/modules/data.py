from os.path import join, pardir, realpath

def get_real_path(*args) -> str:
    return str(realpath(join(*args)))

parent_path = get_real_path(__file__, pardir, pardir, pardir, pardir)

data_path = get_real_path(parent_path, "data")

train_data = [
    get_real_path(data_path, "image.npy"),
    get_real_path(data_path, "label.npy")
]
train_data_full = [
    get_real_path(data_path, "image_full.npy"),
    get_real_path(data_path, "label_full.npy")
]
tests_data = [
    get_real_path(data_path, "tests.npy"),
    get_real_path(data_path, "names.npy")
]

output_path = get_real_path(parent_path, "results")

model_path = get_real_path(output_path, "model_state_dict.pt")
result_path = get_real_path(output_path, "result.csv")
