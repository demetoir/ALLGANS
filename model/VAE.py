from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *


def square_loss(a, b, name='square_loss'):
    with tf.variable_scope(name):
        return tf.pow((a - b), 2)


class VAE(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]

    def load_hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.001
        self.beta1 = 0.5

    def encoder(self, Xs, name='encoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(128, relu)

            h = stack.last_layer
            mean = h[:, :64]
            std = tf.nn.softplus(h[:, 64:])

        return mean, std

    def decoder(self, Xs, name='decoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(128, relu)
            stack.linear_block(256, relu)
            stack.linear_block(784, sigmoid)

        return stack.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, [self.batch_size] + self.X_shape, name='Xs')
        self.Xs_image = tf.reshape(self.Xs, [self.batch_size] + [28, 28], name='Xs_image')
        self.Xs_flatten = tf.reshape(self.Xs, [self.batch_size, -1], name='Xs_flatten')

        self.mean, self.std = self.encoder(self.Xs_flatten)
        self.z = self.mean + self.std * tf.random_normal(self.mean.shape, 0, 1, dtype=tf.float32)

        self.Xs_gen = self.decoder(self.z)
        self.Xs_gen_image = tf.reshape(self.Xs_gen, [self.batch_size] + [28, 28], )

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def load_loss_function(self):
        with tf.variable_scope('loss'):
            X = self.Xs
            X_out = self.Xs_gen
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
