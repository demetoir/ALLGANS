from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *


class AE(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [self.batch_size] + self.X_shape

    def load_hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10

    def encoder(self, Xs, name='encoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(64, relu)

        return stack.last_layer

    def decoder(self, Xs, name='decoder'):
        with tf.variable_scope(name):
            stack = Stacker(Xs)
            stack.linear_block(128, relu)
            stack.linear_block(256, relu)
            stack.linear_block(784, relu)
            stack.linear_block(784, sigmoid)

        return stack.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')

        def is_vector(x):
            if len(x) == 1:
                return True
            else:
                return False

        Xs = self.Xs
        if not is_vector(self.X_shape):
            self.Xs_flatten = flatten(self.Xs)
            Xs = self.Xs_flatten

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
