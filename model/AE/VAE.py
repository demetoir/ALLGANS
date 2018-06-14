from functools import reduce
from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *


class VAE(AbstractModel):
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
        self.learning_rate = 0.001
        self.beta1 = 0.5
        self.z_size = 32

    def encoder(self, Xs, name='encoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.flatten()
            stack.linear_block(512, relu)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(self.z_size * 2, relu)

            h = stack.last_layer
            mean = h[:, :self.z_size]
            std = tf.nn.softplus(h[:, self.z_size:])

        return mean, std

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

        self.mean, self.std = self.encoder(self.Xs)
        self.zs_gen = self.mean + self.std * tf.random_normal(self.mean.shape, 0, 1, dtype=tf.float32)
        self.code = self.zs_gen
        self.Xs_recon = self.decoder(self.zs_gen)
        self.Xs_gen = self.decoder(self.zs, reuse=True)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def load_loss_function(self):
        X = tf.reshape(self.Xs, [self.batch_size, -1])
        X_out = tf.reshape(self.Xs_recon, [self.batch_size, -1])
        mean = self.mean
        std = self.std

        self.cross_entropy = tf.reduce_sum(X * tf.log(X_out) + (1 - X) * tf.log(1 - X_out), axis=1)
        self.KL_Divergence = 0.5 * tf.reduce_sum(
            1 - tf.log(tf.square(std) + 1e-8) + tf.square(mean) + tf.square(std), axis=1)

        # in autoencoder's perspective loss can be divide to reconstruct error and regularization error
        # self.recon_error = -1 * self.cross_entropy
        # self.regularization_error = self.KL_Divergence
        # self.loss = self.recon_error + self.regularization_error

        # only cross entropy loss also work
        # self.loss = -1 * self.cross_entropy

        # using MSE than cross entropy loss also work but slow
        # self.MSE= tf.reduce_sum(tf.squared_difference(X, X_out), axis=1)
        # self.loss = self.MSE + self.KL_Divergence

        # this one also work
        # self.loss = self.MSE

        self.loss = -1 * self.cross_entropy + self.KL_Divergence
        self.loss_mean = tf.reduce_mean(self.loss)

    def load_train_ops(self):
        self.train = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss_mean,
                                                                                     var_list=self.vars)

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key]
        )
        sess.run(
            [self.train, self.op_inc_global_step],
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
