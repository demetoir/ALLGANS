from PIL import Image
from skimage import color, io
import math
import numpy as np


def np_img_float_to_uint8(np_img):
    return np.uint8(np_img * 255)


def np_img_uint8_to_float32(np_img):
    return np.float32(np_img) / 255.0


def np_img_rgb_to_gray(np_img):
    return np.uint8(color.rgb2gray(np_img) * 255)


def np_img_gray_to_rgb(np_img):
    return color.gray2rgb(np_img)


def np_img_to_PIL_img(np_img):
    return Image.fromarray(np_img)


def np_img_from_file(path):
    return io.imread(path)


def np_img_to_tile(np_imgs, column_size=10):
    """rgb out"""
    n, h, w, c = np_imgs.shape
    row_size = int(math.ceil(float(np_imgs.shape[0]) / float(column_size)))
    tile_width = w * column_size
    tile_height = h * row_size

    img_idx = 0
    # HWC format
    tile = np.ones((tile_height, tile_width, 3), dtype=np.uint8)
    for y in range(0, tile_height, h):
        for x in range(0, tile_width, w):
            if img_idx >= n:
                break

            tile[y:y + h, x:x + w, :] = np_imgs[img_idx]
            img_idx += 1

        if img_idx >= n:
            break

    return tile


def np_img_NCWH_to_NHWC(imgs):
    return np.transpose(imgs, [0, 2, 3, 1])


def np_index_to_onehot(x, n=None, dtype=float):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]
