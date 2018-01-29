import tarfile
import zipfile

import numpy as np
import requests
from PIL import Image
from skimage import io
from skimage import color
import math


# pickle util
def dump(path, data):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    del pickle


def load(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    del pickle
    return data


# pillow util
def PIL_img_to_np_img(PIL_img):
    return np.uint8(np.asarray(PIL_img))


def PIL_img_from_file(path):
    return Image.open(path)


# numpy
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


# TODO need relocate
def load_class_from_source_path(module_path, class_name):
    from importlib._bootstrap_external import SourceFileLoader

    module_ = SourceFileLoader('', module_path).load_module()
    return getattr(module_, class_name)


def imports():
    import types
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


# wrapper function

# TODO implement this wrapper
def function_logger(func):
    def wrapper(func):
        ret = func()

        return ret

    return wrapper(func)


def download_data(source_url, download_path):
    r = requests.get(source_url, allow_redirects=True)
    open(download_path, 'wb').write(r.content)


def extract_tar(source_path, destination_path):
    with tarfile.open(source_path) as file:
        file.extractall(destination_path)


def extract_zip(source_path, destination_path):
    with zipfile.ZipFile(source_path) as file:
        file.extractall(destination_path)


def extract_data(source_path, destination_path):
    extender_tar = ['tar.gz', 'tar', 'gz']
    extender_zip = ['zip']

    extend = source_path.split('.')[-1]
    if extend in extender_tar:
        extract_tar(source_path, destination_path)
    elif extend in extender_zip:
        extract_zip(source_path, destination_path)
