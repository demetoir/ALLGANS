from functools import reduce
from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
import numpy as np


class AAE(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [self.batch_size] + self.X_shape
        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.Y_batch_key = 'Ys'
        self.Y_shape = input_shapes[self.Y_batch_key]
        self.Ys_shape = [self.batch_size] + self.Y_shape
        self.Y_size = self.Y_shape[0]

        self.z_shape = [self.z_size]
        self.zs_shape = [self.batch_size] + self.z_shape

    def load_hyper_parameter(self, params=None):
        self.batch_size = 100
        self.learning_rate = 0.001
        self.beta1 = 0.5
        self.z_size = 10

    def encoder(self, Xs, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()
            stack.linear_block(512, relu)
            stack.linear_block(256, relu)
            stack.linear_block(128, relu)
            stack.linear_block(self.z_size + self.Y_size, relu)
            zs = stack.last_layer[:, :self.z_size]
            Ys_gen = stack.last_layer[:, self.z_size:]

            hs = softmax(Ys_gen)
        return zs, Ys_gen, hs

    def decoder(self, zs, Ys, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((zs, Ys), axis=1))
            stack.linear_block(128, relu)
            stack.linear_block(256, relu)
            stack.linear_block(512, relu)
            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    def discriminator_gauss(self, zs, reuse=False, name='discriminator_gauss'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
            layer.linear_block(256, relu)
            layer.linear_block(256, relu)
            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def discriminator_cate(self, zs, reuse=False, name='discriminator_cate'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
            layer.linear_block(256, relu)
            layer.linear_block(256, relu)
            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = tf.placeholder(tf.float32, self.zs_shape, name='zs')

        self.zs_gen, self.Ys_gen, self.hs = self.encoder(self.Xs)
        self.code = self.zs_gen
        self.Xs_recon = self.decoder(self.zs_gen, self.Ys_gen)
        self.Xs_gen = self.decoder(self.zs, self.Ys, reuse=True)

        self.D_gauss_real = self.discriminator_gauss(self.zs)
        self.D_gauss_gen = self.discriminator_gauss(self.zs_gen, reuse=True)

        self.D_cate_real = self.discriminator_cate(self.Ys)
        self.D_cate_gen = self.discriminator_cate(self.Ys_gen, reuse=True)

        self.vars_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        self.vars_discriminator_gauss = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_gauss')
        self.vars_discriminator_cate = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_cate')

        self.predict = tf.equal(tf.arg_max(self.hs, 1), tf.arg_max(self.Ys, 1), name='predict')
        self.acc = tf.reduce_mean(tf.cast(self.predict, tf.float32), name='acc')

    def load_loss_function(self):
        # AE loss
        self.loss_AE = tf.squared_difference(self.Xs, self.Xs_recon, name="loss_AE")
        self.loss_AE_mean = tf.reduce_sum(self.loss_AE, name="loss_AE_mean")

        # D gauss loss
        self.loss_D_gauss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_gauss_real),
                                                                         logits=self.D_gauss_real,
                                                                         name='loss_D_gauss_real')
        self.loss_D_gauss_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_gauss_gen),
                                                                        logits=self.D_gauss_gen,
                                                                        name='loss_D_gauss_gen')

        self.loss_D_gauss = tf.add(self.loss_D_gauss_real, self.loss_D_gauss_gen, name='loss_D_gauss')
        self.loss_D_gauss_mean = tf.reduce_mean(self.loss_D_gauss, name='loss_D_gauss_mean')

        # D cate loss
        self.loss_D_cate_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_cate_real),
                                                                        logits=self.D_cate_real,
                                                                        name='loss_D_cate_real')
        self.loss_D_cate_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_cate_gen),
                                                                       logits=self.D_cate_gen,
                                                                       name='loss_D_cate_gen')
        self.loss_D_cate = tf.add(self.loss_D_cate_real, self.D_cate_gen, name='loss_D_cate')
        self.loss_D_cate_mean = tf.reduce_mean(self.loss_D_cate, name='loss_D_cate_mean')

        # G gauss loss
        self.loss_G_gauss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_gauss_gen),
                                                                    logits=self.D_gauss_gen,
                                                                    name='loss_G_gauss')
        self.loss_G_gauss_mean = tf.reduce_mean(self.loss_G_gauss, name='loss_G_gauss_mean')

        # G cate loss
        self.loss_G_cate = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_cate_gen),
                                                                   logits=self.D_cate_gen,
                                                                   name='loss_G_cate')
        self.loss_G_cate_mean = tf.reduce_mean(self.loss_G_cate, name='loss_G_cate_mean')

        # classifier phase
        self.loss_clf = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys,
                                                                   logits=self.hs,
                                                                   name='loss_clf')
        self.loss_clf_mean = tf.reduce_mean(self.loss_clf, name='loss_clf_mean')

        # in autoencoder's perspective loss can be divide to reconstruct error and regularization error
        # self.recon_error = -1 * self.cross_entropy
        # self.regularization_error = self.KL_Divergence
        # self.loss = self.recon_error + self.regularization_error

        # only cross entropy loss also work
        # self.loss = -1 * self.cross_entropy

        # using MSE than cross entropy loss also work but slow
        # self.MSE= tf.reduce_sum(tf.squared_difference(X, X_gen), axis=1)
        # self.loss = self.MSE + self.KL_Divergence

        # this one also work
        # self.loss = self.MSE

    def load_train_ops(self):
        # reconstruction phase
        self.train_AE = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_AE_mean,
            var_list=self.vars_decoder + self.vars_encoder
        )

        # regularization phase
        self.train_D_gauss = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_D_gauss,
            var_list=self.vars_discriminator_gauss
        )

        self.train_D_cate = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_D_cate,
            var_list=self.vars_discriminator_cate
        )

        self.train_G_gauss = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_gauss,
            var_list=self.vars_encoder
        )

        self.train_G_cate = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_G_cate,
            var_list=self.vars_encoder
        )

        self.train_clf = tf.train.AdamOptimizer(
            self.learning_rate,
            self.beta1
        ).minimize(
            self.loss_clf,
            var_list=self.vars_encoder

        )

    def train_model(self, sess=None, iter_num=None, dataset=None):
        Xs, Ys = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key]
        )
        sess.run(
            [self.train_AE, self.train_D_gauss, self.train_D_cate, self.train_G_gauss, self.train_G_cate,
             self.train_clf, self.op_inc_global_step],
            feed_dict={
                self.Xs: Xs,
                self.Ys: Ys,
                self.zs: self.get_z_noise()
            }
        )

    def load_summary_ops(self):
        pass

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        pass

    def get_z_noise(self):
        return np.random.uniform(-1.0, 1.0, size=self.zs_shape)

    def run(self, sess, fetches, dataset):
        Xs, Ys = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key],
            look_up=True
        )
        return self.get_tf_values(sess, fetches, Xs, Ys, self.get_z_noise())

    def get_tf_values(self, sess, fetches, Xs=None, Ys=None, zs=None):
        return sess.run(
            fetches=fetches,
            feed_dict={
                self.Xs: Xs,
                self.Ys: Ys,
                self.zs: zs
            }
        )
