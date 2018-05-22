from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
from functools import reduce


class AE(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [self.batch_size] + self.X_shape
        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)
        self.z_shape = [self.z_size]
        self.zs_shape = [self.batch_size] + self.z_shape

    def load_hyper_parameter(self, params=None):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10
        self.code_size = 32
        self.z_size = 32

    def encoder(self, Xs, name='encoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.flatten()
            stack.linear_block(512, relu)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(self.code_size, relu)

        return stack.last_layer

    def decoder(self, zs, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(zs)
            stack.linear_block(128, relu)
            stack.linear_block(256, relu)
            stack.linear_block(512, relu)
            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = tf.placeholder(tf.float32, self.zs_shape, name='zs')

        self.code = self.encoder(self.Xs)
        self.Xs_recon = self.decoder(self.code)
        self.Xs_gen = self.decoder(self.zs, reuse=True)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def load_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self.Xs_recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def load_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key]
        )
        sess.run(
            [self.train_op, self.op_inc_global_step],
            feed_dict={
                self.Xs: batch_xs,
            }
        )

    def load_summary_ops(self):
        pass

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass

    def run(self, sess, fetches, dataset):
        batch_xs = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key],
            look_up=True
        )
        return sess.run(
            fetches=fetches,
            feed_dict={
                self.Xs: batch_xs,
            }
        )

    def get_tf_values(self, sess, fetches, Xs):
        return sess.run(
            fetches,
            feed_dict={
                self.Xs: Xs
            })
