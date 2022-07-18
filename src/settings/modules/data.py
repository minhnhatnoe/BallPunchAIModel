from os.path import join, pardir, realpath

def get_real_path(*args) -> str:
    return str(realpath(join(*args)))

parent_path = get_real_path(__file__, pardir, pardir, pardir, pardir)

class Input:
    data_path = get_real_path(parent_path, "data")

    x_path = get_real_path(data_path, "image.npy")
    y_path = get_real_path(data_path, "label.npy")
    t_path = get_real_path(data_path, "tests.npy")
    n_path = get_real_path(data_path, "names.npy")

class Output:
    result_path = get_real_path(parent_path, "result")

    model_path = get_real_path(result_path, "model_state_dict.pt")
    result_path = get_real_path(result_path, "result.csv")
