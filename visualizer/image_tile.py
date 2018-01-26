import os

from visualizer.AbstractVisualizer import AbstractVisualizer
from util.util import *
import numpy as np


class image_tile(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs0 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs1 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs = np.concatenate((sample_imgs0, sample_imgs1))
        sample_imgs = np.uint8(sample_imgs * 255)

        img_path = os.path.join(self.visualizer_path, '{}.png'.format(str(iter_num).zfill(5)))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        pil_img = np_img_to_PIL_img(tile)
        with open(img_path, 'wb') as fp:
            pil_img.save(fp)
