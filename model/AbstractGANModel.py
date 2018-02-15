from model.AbstractModel import AbstractModel
from dict_keys.input_shape_keys import *


class AbstractGANModel(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __init__(self, logger_path):
        super().__init__(logger_path)

    def load_input_shapes(self, input_shapes):
        shape_data_x = input_shapes[INPUT_SHAPE_KEY_DATA_X]
        if len(shape_data_x) == 3:
            self.shape_data_x = shape_data_x
            H, W, C = shape_data_x
            self.input_size = W * H * C
            self.input_w = W
            self.input_h = H
            self.input_c = C
        elif len(shape_data_x) == 2:
            self.shape_data_x = shape_data_x + [1]
            H, W = shape_data_x
            self.input_size = W * H
            self.input_w = W
            self.input_h = H
            self.input_c = 1

    def load_hyper_parameter(self):
        raise NotImplementedError

    def load_main_tensor_graph(self):
        raise NotImplementedError

    def load_loss_function(self):
        raise NotImplementedError

    def load_train_ops(self):
        raise NotImplementedError

    def train_model(self, sess=None, iter_num=None, dataset=None):
        raise NotImplementedError

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        raise NotImplementedError

    def load_summary_ops(self):
        raise NotImplementedError
