from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
from dict_keys.dataset_batch_keys import *


class image_tile_data(AbstractVisualizer):
    """visualizer tile image from dataset images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X])
        if sample_imgs.dtype == np.dtype("float32"):
            sample_imgs = np_img_float32_to_uint8(sample_imgs)

        file_name = '{}.png'.format(str(iter_num).zfill(5))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        self.save_np_img(tile, file_name)
