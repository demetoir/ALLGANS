from model.AbstractModel import AbstractModel
import tensorflow as tf


class DummyModel(AbstractModel):

    def load_input_shapes(self, input_shapes):
        pass

    def __init__(self, logger_path=None):
        super().__init__(logger_path)
        self.iter_count = 0

    def load_train_ops(self):
        pass

    def load_hyper_parameter(self, params=None):
        self.batch_size = 1

        pass

    def load_main_tensor_graph(self):
        self.v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
        self.v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

        self.inc_v1 = self.v1.assign(self.v1 + 1)
        self.dec_v2 = self.v2.assign(self.v2 - 1)

    def load_loss_function(self):
        pass

    def load_summary_ops(self):
        pass

    def train_model(self, sess=None, iter_num=None, dataset=None):
        self.iter_count += 1
        sess.run([self.inc_v1, self.dec_v2])

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass
