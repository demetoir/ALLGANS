from model.sklearn_like_model.AE.BaseAutoEncoder import BaseAutoEncoder
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
            'encoder_net_shapes',
            'decoder_net_shapes'
        ]

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _zs(self):
        return self.zs

    @property
    def _train_ops(self):
        return [self.train_op, self.op_inc_global_step]

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
        self.latent_code_size = 32
        self.z_size = 32
        self.encoder_net_shapes = [512, 256, 128]
        self.decoder_net_shapes = [128, 256, 512]

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape

        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.z_shape = [self.z_size]
        self.zs_shape = [None] + self.z_shape

    def encoder(self, Xs, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()
            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.latent_code_size, relu)

        return stack.last_layer

    def decoder(self, zs, net_shapes, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(zs)
            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    def build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')

        self.latent_code = self.encoder(self.Xs, self.encoder_net_shapes)
        self.Xs_recon = self.decoder(self.latent_code, self.decoder_net_shapes)
        self.Xs_gen = self.decoder(self.zs, self.decoder_net_shapes, reuse=True)

        head = get_scope()
        self.vars = collect_vars(join_scope(head, 'encoder'))
        self.vars += collect_vars(join_scope(head, 'decoder'))

    def build_loss_function(self):
        self.loss = tf.squared_difference(self.Xs, self.Xs_recon, name='loss')
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(loss=self.loss,
                                                                                        var_list=self.vars)

    def random_z(self):
        pass
