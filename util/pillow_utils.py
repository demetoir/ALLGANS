"""pillow util"""
import numpy as np
from PIL import Image


def PIL_img_to_np_img(PIL_img):
    return np.uint8(np.asarray(PIL_img))


def PIL_img_from_file(path):
    return Image.open(path)
