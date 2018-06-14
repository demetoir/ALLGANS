from model.AbstractModel import AbstractModel
from dict_keys.input_shape_keys import *


class AbstractGANModel(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __init__(self, logger_path):
        super().__init__(logger_path)

    def load_input_shapes(self, input_shapes):
        X_shape = input_shapes[INPUT_SHAPE_KEY_DATA_X]
        if len(X_shape) == 3:
            self.X_shape = X_shape
            H, W, C = X_shape
            self.input_size = W * H * C
            self.input_w = W
            self.input_h = H
            self.input_c = C
        elif len(X_shape) == 2:
            self.X_shape = X_shape + [1]
            H, W = X_shape
            self.input_size = W * H
            self.input_w = W
            self.input_h = H
            self.input_c = 1

    def load_hyper_parameter(self, params=None):
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
