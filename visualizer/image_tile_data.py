from util.numpy_utils import *
from visualizer.AbstractVisualizer import AbstractVisualizer
from dict_keys.dataset_batch_keys import *
import os


class image_tile_data(AbstractVisualizer):
    """visualizer tile image from dataset images"""
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        sample_imgs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X])

        img_path = os.path.join(self.visualizer_path, '{}.png'.format(str(iter_num).zfill(5)))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        pil_img = np_img_to_PIL_img(tile)
        with open(img_path, 'wb') as fp:
            pil_img.save(fp)
