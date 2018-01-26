from data_handler.LLD import LLD
from env_settting import LLD_PATH
from dict_keys.input_shape_keys import *


class LLD_helper:
    @staticmethod
    def next_batch_task(batch):
        x = batch[0]
        return x

    @staticmethod
    def load_dataset(limit=None):
        lld_data = LLD(batch_after_task=LLD_helper.next_batch_task)
        lld_data.load(LLD_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 3],
        }

        return lld_data, input_shapes
