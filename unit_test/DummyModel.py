from model.AbstractModel import AbstractModel
import tensorflow as tf


class DummyModel(AbstractModel):

    def input_shapes(self, input_shapes):
        pass

    def __init__(self, metadata, input_shapes):
        super().__init__(metadata, input_shapes)
        self.iter_count = 0

    def train_ops(self):
        pass

    def hyper_parameter(self):
        self.batch_size = 1

        pass

    def network(self):
        self.v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
        self.v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

        self.inc_v1 = self.v1.assign(self.v1 + 1)
        self.dec_v2 = self.v2.assign(self.v2 - 1)

    def train(self, sess=None, iter_num=None, dataset=None):
        self.iter_count += 1
        sess.run([self.inc_v1, self.dec_v2])

    def loss(self):
        pass

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass

    def summary_ops(self):
        pass
