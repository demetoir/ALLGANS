from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
import numpy as np


class DAE(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [self.batch_size] + self.X_shape

        self.noise_shape = self.Xs_shape

    def load_hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10
        self.code_size = 32

    def encoder(self, Xs, name='encoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(self.code_size, relu)

        return stack.last_layer

    def decoder(self, Xs, name='decoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(128, relu)
            stack.linear_block(256, relu)
            stack.linear_block(self.Xs_flatten_size, relu)
            stack.linear_block(self.Xs_flatten_size, sigmoid)

        return stack.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.noise = tf.placeholder(tf.float32, self.noise_shape, name='noise')

        def is_vector(x):
            if len(x) == 1:
                return True
            else:
                return False

        self.Xs_noised = tf.add(self.Xs, self.noise, name='Xs_noised')

        Xs = self.Xs_noised
        if not is_vector(self.X_shape):
            self.Xs_flatten = flatten(Xs)
            Xs = self.Xs_flatten
            self.Xs_flatten_size = self.Xs_flatten.shape[1]

        self.code = self.encoder(Xs)
        self.Xs_gen = self.decoder(self.code)
        self.Xs_gen = reshape(self.Xs_gen, self.Xs_shape)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def load_loss_function(self):
        with tf.variable_scope('loss'):
            self.loss = tf.squared_difference(self.Xs, self.Xs_gen, name='loss')
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
                self.noise: self.get_noise()
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
                self.noise: self.get_noise()
            }
        )

    def get_noise(self):
        return np.random.normal(-1, 1, size=self.Xs_shape)
