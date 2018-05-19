from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class image_AAE_Ys(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        Xs = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=[model.X_batch_key],
            look_up=True
        )
        Ys = np.array(10 * [i for i in range(10)])
        Ys = np_index_to_onehot(Ys)

        Xs_recon, global_step = model.get_tf_values(
            sess,
            [model.Xs_recon, model.global_step],
            Xs, Ys, model.get_z_noise(),
        )
        Xs_recon_np_img = np_img_float32_to_uint8(Xs_recon)

        Xs_real = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=['Xs'],
            look_up=True
        )
        Xs_real_np_img = np_img_float32_to_uint8(Xs_real)
        concat_np_img = np.concatenate((Xs_real_np_img, Xs_recon_np_img), axis=0)

        # sample_imgs = Xs_gen
        file_name = '{}.png'.format(str(iter_num).zfill(8))
        tile = np_img_to_tile(concat_np_img, column_size=10)
        self.save_np_img(tile, file_name)
