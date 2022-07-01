from os import path
import cv2
import numpy as np

standard_dim = (256, 144)

def temp_x_path() -> str:
    return str(path.realpath(path.join(
        path.realpath(__file__),
        path.pardir,
        "temp_x.npy"
    )))
def temp_y_path() -> str:
    return str(path.realpath(path.join(
        path.realpath(__file__),
        path.pardir,
        "temp_y.npy"
    )))


def correct_image(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, standard_dim)


def get_image_file(path: str) -> np.ndarray:
    image = cv2.imread(path)
    return correct_image(image)


def show_image(image: np.ndarray) -> None:
    cv2.imshow("", image)
    cv2.waitKey()
