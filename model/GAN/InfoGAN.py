from model.AbstractGANModel import AbstractGANModel
from util.Stacker import Stacker
from util.tensor_ops import *
import numpy as np
import tensorflow as tf


class InfoGAN(AbstractGANModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.Y_batch_key = 'Ys'
        X_shape = input_shapes[self.X_batch_key]
        self.Y_shape = input_shapes[self.Y_batch_key]
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
        self.Ys_shape = [self.batch_size] + self.Y_shape
        self.z_shape = [self.n_noise]
        self.zs_shape = [self.batch_size] + self.z_shape
        self.c_shape = [self.n_c]
        self.cs_shape = [self.batch_size] + self.c_shape

    def load_hyper_parameter(self, params=None):
        self.n_noise = 256
        self.n_c = 2
        self.batch_size = 64
        self.learning_rate = 0.0002

        self.len_discrete_code = 10  # categorical distribution (i.e. label)
        self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)

    def Q_function(self, X_gen, reuse=False):
        with tf.variable_scope('Q_function', reuse=reuse):
            layer = Stacker(X_gen)
            layer.linear_block(128, relu)
            layer.linear_block(128, relu)
            code_logit = layer.linear(10 + 2)
            code = layer.softmax()

        return code, code_logit

    def generator(self, z, y, c, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            layer = Stacker(concat((z, y, c), axis=1))

            layer.add_layer(linear, 7 * 7 * 128)
            layer.reshape([self.batch_size, 7, 7, 128])
            layer.upscale_2x_block(256, CONV_FILTER_5522, relu)
            layer.conv2d_transpose(self.Xs_shape, CONV_FILTER_5522)
            layer.conv2d(self.input_c, CONV_FILTER_3311)
            layer.sigmoid()

        return layer.last_layer

    def discriminator(self, X, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            layer = Stacker(X)

            layer.conv_block(128, CONV_FILTER_5522, lrelu)
            layer.conv_block(256, CONV_FILTER_5522, lrelu)
            net = layer.reshape([self.batch_size, -1])
            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer, net

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = tf.placeholder(tf.float32, self.zs_shape, name='zs')
        self.cs = tf.placeholder(tf.float32, self.cs_shape, name='cs')

        self.D_real, input4classifier_real = self.discriminator(self.Xs)

        self.Xs_gen = self.generator(self.zs, self.Ys, self.cs)
        self.G = self.Xs_gen
        self.D_gen, D_feature_gen = self.discriminator(self.Xs_gen, True)
        self.code, self.code_logit = self.Q_function(D_feature_gen)

    def load_loss_function(self):
        # # get loss for discriminator
        # d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        # d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        self.loss_D_real = tf.reduce_mean(self.D_real, name='loss_D_real')

        self.loss_D_gen = tf.reduce_mean(self.D_gen, name='loss_D_gen')

        self.loss_D = - tf.reduce_mean(tf.log(self.D_real)) - tf.reduce_mean(tf.log(1. - self.D_gen), name='loss_D')

        self.loss_G = - tf.reduce_mean(tf.log(self.D_gen), name='loss_G')

        # discrete code : categorical
        disc_code_est = self.code_logit[:, :self.len_discrete_code]
        disc_code_tg = self.Ys
        Q_disc_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg))

        # continuous code : gaussian
        cont_code_est = self.code_logit[:, self.len_discrete_code:]
        cont_code_tg = self.cs
        Q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))

        # get information loss
        self.loss_Q = Q_disc_loss + Q_cont_loss
        # self.loss_Q = Q_disc_loss

    def load_train_ops(self):
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='discriminator')
        self.train_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss_D, var_list=self.vars_D)

        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='generator')
        self.train_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss_G, var_list=self.vars_G)

        self.vars_Q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_function')
        var_list = self.vars_D + self.vars_G + self.vars_Q
        self.train_Q = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss_Q, var_list=var_list)

    def load_misc_ops(self):
        super().load_misc_ops()

    def train_model(self, sess=None, iter_num=None, dataset=None):
        Xs, Ys = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key])
        sess.run(
            [self.train_G, self.train_D, self.train_Q, self.op_inc_global_step],
            feed_dict={
                self.Xs: Xs,
                self.zs: self.get_z_noise(),
                self.Ys: Ys,
                self.cs: self.get_c_noise(),
            }
        )

    def run(self, sess, fetches, dataset):
        Xs, Ys = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key])
        return sess.run(
            fetches,
            feed_dict={
                self.Xs: Xs,
                self.zs: self.get_z_noise(),
                self.Ys: Ys,
                self.cs: self.get_c_noise(),
            }
        )

    def get_z_noise(self):
        return np.random.uniform(-1.0, 1.0, size=self.zs_shape)

    def get_c_noise(self):
        return np.random.uniform(-1.0, 1.0, size=self.cs_shape)

    def load_summary_ops(self):
        pass

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass
