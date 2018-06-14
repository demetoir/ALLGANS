from model.AbstractGANModel import AbstractGANModel
from util.Stacker import Stacker
from util.tensor_ops import *
import numpy as np
import tensorflow as tf


class GAN(AbstractGANModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        X_shape = input_shapes[self.X_batch_key]
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
        self.Xs_shape = [self.batch_size] + self.X_shape
        self.z_shape = [self.n_noise]
        self.zs_shape = [self.batch_size] + self.z_shape

    def load_hyper_parameter(self, params=None):
        self.n_noise = 256
        self.batch_size = 64
        self.learning_rate = 0.0002

    def generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            layer = Stacker(z)
            layer.add_layer(linear, 7 * 7 * 128)
            layer.reshape([self.batch_size, 7, 7, 128])
            layer.upscale_2x_block(256, CONV_FILTER_5522, relu)
            layer.conv2d_transpose(self.Xs_shape, CONV_FILTER_5522)
            layer.conv2d(self.input_c, CONV_FILTER_3311)
            layer.sigmoid()

        return layer.last_layer

    def discriminator(self, X, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = Stacker(X)
            layer.conv_block(128, CONV_FILTER_5522, lrelu)
            layer.conv_block(256, CONV_FILTER_5522, lrelu)
            layer.reshape([self.batch_size, -1])
            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name="Xs")
        self.zs = tf.placeholder(tf.float32, self.zs_shape, name="zs")

        self.G = self.generator(self.zs)
        self.Xs_gen = self.G
        self.D_real = self.discriminator(self.Xs)
        self.D_gen = self.discriminator(self.Xs_gen, reuse=True)

    def load_loss_function(self):
        self.loss_D_real = tf.reduce_mean(self.D_real, name='loss_D_real')
        self.loss_D_gen = tf.reduce_mean(self.D_gen, name='loss_D_gen')

        self.loss_D = tf.reduce_mean(-tf.log(self.D_real) - tf.log(1. - self.D_gen), name='loss_D')
        self.loss_G = tf.reduce_mean(-tf.log(self.D_gen), name='loss_G')

    def load_train_ops(self):
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='discriminator')
        self.train_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss_D, var_list=self.vars_D)

        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='generator')
        self.train_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss_G, var_list=self.vars_G)

    def load_misc_ops(self):
        super().load_misc_ops()

    def train_model(self, sess=None, iter_num=None, dataset=None):
        Xs = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key]
        )
        sess.run(
            [self.train_G, self.train_D, self.op_inc_global_step],
            feed_dict={
                self.Xs: Xs,
                self.zs: self.get_noise()
            }
        )

    def run(self, sess, fetches, dataset):
        Xs = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key],
            look_up=True
        )
        return sess.run(
            fetches,
            feed_dict={
                self.Xs: Xs,
                self.zs: self.get_noise()
            }
        )

    def get_noise(self):
        return np.random.uniform(-1.0, 1.0, size=self.zs_shape)

    def load_summary_ops(self):
        pass

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass
