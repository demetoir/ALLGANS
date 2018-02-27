from data_handler.LLD import LLD
from env_settting import *
from util.numpy_utils import np_img_rgb_to_gray, np_img_gray_to_rgb, np_img_to_PIL_img, np_img_load_file, \
    np_img_to_tile
import numpy as np
import cv2
import os


decomposed_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_decomposed')
if not os.path.exists(decomposed_path):
    os.mkdir(decomposed_path)

packed_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_packed')
if not os.path.exists(packed_path):
    os.mkdir(packed_path)
packed_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_packed', 'packed.pkl')


def edge_tile():
    img_list = []
    for img_number in range(1, 50 + 1):
        img_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_decomposed', '0000', str(img_number).zfill(3) + '.jpg')
        np_img_rgb = np_img_load_file(img_path)

        img = cv2.imread(img_path, 0)
        edges0 = np_img_gray_to_rgb(cv2.Canny(img, 100, 200))
        edges1 = np_img_gray_to_rgb(cv2.Canny(img, 50, 200))
        edges2 = np_img_gray_to_rgb(cv2.Canny(img, 100, 300))
        img = np_img_gray_to_rgb(img)
        img_list += [np_img_rgb, img, edges0, edges1, edges2]

    img_list = np.array(img_list)
    tile = np_img_to_tile(img_list, column_size=5)
    img = np_img_to_PIL_img(tile)
    img.show()


def batch_process(img):
    for _ in range(100000):
        np_img = np_img_gray_to_rgb(img)


def extract_edge(batch):
    ret = np.zeros(shape=[batch.shape[0], 32, 32, 1])

    for i in range(batch.shape[0]):
        data = np_img_rgb_to_gray(batch[i])
        data = cv2.Canny(data, 100, 200)
        ret[i] = np.reshape(data, [32, 32, 1])

    return ret


if __name__ == '__main__':
    LLD_PATH = os.path.join(ROOT_PATH, 'dataset', 'LLD')
    lld_data = LLD(extract_edge)

    lld_data.load(LLD_PATH)

    rgb = lld_data.next_batch(10, lookup=True)
    edge = lld_data.next_batch(10, lookup=True)
    lld_data.next_batch(10)
    edge = np_img_gray_to_rgb(edge)
    img = np.concatenate((rgb, edge))
    for i in range(5):
        rgb = lld_data.next_batch(10, lookup=True)
        edge = lld_data.next_batch(10, lookup=True)
        lld_data.next_batch(10)
        edge = np_img_gray_to_rgb(edge)
        img = np.concatenate((img, rgb, edge))

    tile = np_img_to_tile(img)
    tile = np_img_to_PIL_img(tile)
    tile.show()
