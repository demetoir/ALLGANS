from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class image_tile(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs0 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs1 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs = np.concatenate((sample_imgs0, sample_imgs1))
        sample_imgs = np_img_float32_to_uint8(sample_imgs)

        file_name = '{}.png'.format(str(iter_num).zfill(8))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        self.save_np_img(tile, file_name)

