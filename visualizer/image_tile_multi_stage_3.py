import os

from visualizer.AbstractVisualizer import AbstractVisualizer
from util.util import *
import numpy as np


class image_tile_multi_stage_3(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        noise = model.get_noise()
        G1_imgs = sess.run(model.G1, feed_dict={model.z: noise})
        G2_imgs = sess.run(model.G2, feed_dict={model.z: noise})
        G3_imgs = sess.run(model.G3, feed_dict={model.z: noise})
        sample_imgs = np.concatenate((G1_imgs, G2_imgs, G3_imgs))
        sample_imgs = np.uint8(sample_imgs * 255)

        img_path = os.path.join(self.visualizer_path, '{}.png'.format(str(iter_num).zfill(5)))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        pil_img = np_img_to_PIL_img(tile)
        with open(img_path, 'wb') as fp:
            pil_img.save(fp)
