from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class image_AE(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        Xs_recon, global_step = model.run(
            sess,
            [model.Xs_recon, model.global_step],
            dataset
        )
        sample_imgs = np_img_float32_to_uint8(Xs_recon)

        batch_xs = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=['Xs'],
            look_up=True
        )
        batch_xs = np_img_float32_to_uint8(batch_xs)
        sample_imgs = np.concatenate((batch_xs, sample_imgs), axis=0)

        # sample_imgs = Xs_recon
        file_name = '{}.png'.format(str(iter_num).zfill(8))
        tile = np_img_to_tile(sample_imgs, column_size=10)
        self.save_np_img(tile, file_name)
