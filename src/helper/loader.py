from typing import List
import numpy as np


def load_data(paths: 'List[str, str]', *args, **kwargs) -> 'List[np.ndarray, np.ndarray]':
    data = [np.load(path, *args, **kwargs) for path in paths]
    for array in data:
        assert(array.shape[0] == data[0].shape[0])
    return data
