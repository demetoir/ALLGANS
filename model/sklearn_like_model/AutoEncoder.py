from model.sklearn_like_model.BaseAutoEncoder import BaseAutoEncoder
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
from functools import reduce


class AutoEncoder(BaseAutoEncoder):

    @property
    def hyper_param_key(self):
        return [
            'batch_size',
            'learning_rate',
            'beta1',
            'L1_norm_lambda',
            'K_average_top_k_loss',
            'code_size',
            'z_size',
        ]

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _zs(self):
        return self.zs

    @property
    def _train_ops(self):
        return [self.train_op]

    @property
    def _code_ops(self):
        return self.latent_code

    @property
    def _recon_ops(self):
        return self.Xs_recon

    @property
    def _generate_ops(self):
        return self.Xs_gen

    @property
    def _metric_ops(self):
        return self.loss

    def hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.L1_norm_lambda = 0.001
        self.K_average_top_k_loss = 10
        self.code_size = 32
        self.z_size = 32

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape

        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.z_shape = [self.z_size]
        self.zs_shape = [None] + self.z_shape

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
            a = stack.last_layer
            print(a.shape, self.Xs_shape)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    def build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = tf.placeholder(tf.float32, self.zs_shape, name='zs')

        self.latent_code = self.encoder(self.Xs)
        self.Xs_recon = self.decoder(self.latent_code)
        self.Xs_gen = self.decoder(self.zs, reuse=True)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    def build_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self.Xs_recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def random_z(self):
        pass
