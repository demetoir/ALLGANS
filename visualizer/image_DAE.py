from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class image_DAE(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        Xs_noised, Xs_recon, global_step = model.run(
            sess,
            [model.Xs_noised, model.Xs_recon, model.global_step],
            dataset
        )
        Xs_gen_img = np_img_float32_to_uint8(Xs_recon)
        Xs_noised_img = np_img_float32_to_uint8(Xs_noised)

        Xs_real = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=['Xs'],
            look_up=True
        )
        Xs_real_img = np_img_float32_to_uint8(Xs_real)
        sample_imgs = np.concatenate((Xs_real_img, Xs_noised_img, Xs_gen_img), axis=0)

        # sample_imgs = Xs_recon
        file_name = '{}.png'.format(str(iter_num).zfill(8))
        tile = np_img_to_tile(sample_imgs, column_size=10)
        self.save_np_img(tile, file_name)
