from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer


class image_tile(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs = model.run(sess, model.G, dataset)
        sample_imgs = np_img_float32_to_uint8(sample_imgs)

        file_name = '{}.png'.format(str(iter_num).zfill(8))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        self.save_np_img(tile, file_name)
