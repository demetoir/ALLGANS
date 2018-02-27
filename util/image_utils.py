"""image util
PILLOW, ...
"""
from glob import glob
from PIL import Image
import numpy as np
import os
import imageio


def PIL_img_to_np_img(PIL_img):
    """convert PIL image to uint8 numpy image"""
    return np.uint8(np.asarray(PIL_img))


def PIL_img_from_file(path):
    """read PIL image from file"""
    return Image.open(path)


def GIF_maker(path, output_file_name="out.gif", interval=None, time_per_frame=0.1):
    """make GIF image from images in path

    * order of GIF frame is images file's name

    :type path: str
    :type output_file_name: str
    :type interval: int
    :type time_per_frame: float
    :param path: source image path
    :param output_file_name: output GIF file name
    default "output.gif"
    :param interval: interval of images to make GIF
    only interval of image file will
    default None,
    if interval is None, all image in path contain in result GIF image
    :param time_per_frame: time per a GIF frame
    default 0.1
    """
    print("GIF maker\n"
          "path: %s\n"
          "interval: %s\n"
          "time per frame: %s\n" % (path, interval, time_per_frame))
    print("collect files")
    files = sorted(glob(os.path.join(path, "*")))

    if interval is not None:
        files_ = []
        for i in range(0, len(files), interval):
            files_ += [files[i]]
        files = files_
    print("total %s files colleted" % (len(files)))

    print("load images")
    images = [imageio.imread(file) for file in files]

    print("save GIF at %s" % os.path.join(os.path.curdir, output_file_name))
    imageio.mimsave(output_file_name, images, duration=time_per_frame)
