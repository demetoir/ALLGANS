from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np
import cv2


def extract_edge(imgs, threshold1=100, threshold2=200):
    ret = np.zeros(shape=[imgs.shape[0], 32, 32, 1])

    for i in range(imgs.shape[0]):
        data = np_img_rgb_to_gray(imgs[i])
        data = cv2.Canny(data, threshold1, threshold2)
        ret[i] = np.reshape(data, [32, 32, 1])

    return ret


class image_edge_tile(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs0 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs1 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs = np.concatenate((sample_imgs0, sample_imgs1))
        sample_imgs = np_img_float32_to_uint8(sample_imgs)

        threshold_list = []
        for i in range(200, 250, 10):
            threshold_list += [(i, 250)]

        for threshold1, threshold2 in threshold_list:
            edge_image = extract_edge(sample_imgs, threshold1, threshold2)

            file_name = 'edge_tile_(%d, %d).png' % (threshold1, threshold2)
            tile = np_img_to_tile(edge_image, column_size=8)
            self.save_np_img(tile, file_name)
