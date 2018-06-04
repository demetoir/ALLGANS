from tqdm import tqdm
from functools import reduce
from model.sklearn_like_model.AE.BaseAutoEncoder import BaseAutoEncoder
from model.sklearn_like_model.DummyDataset import DummyDataset
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
import numpy as np


class DAE(BaseAutoEncoder):

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
            'decoder_net_shapes',
            'noise_intensity'
        ]

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _noise(self):
        return self.noise

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
        self.noise_intensity = 1.

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape

        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.z_shape = [self.z_size]
        self.zs_shape = [None] + self.z_shape

        self.noise_shape = [None] + self.X_shape

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
        self.noise = placeholder(tf.float32, self.noise_shape, name='noise')

        self.Xs_noised = tf.add(self.Xs, self.noise, name='Xs_noised')

        self.latent_code = self.encoder(self.Xs_noised, self.encoder_net_shapes)
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

    def get_noise(self, shape=None):
        if shape is None:
            shape = self.Xs_shape
        return np.random.normal(-1 * self.noise_intensity, 1 * self.noise_intensity, size=shape)

    def train(self, Xs, epoch=100, save_interval=None, batch_size=None):
        self.if_not_ready_to_train()

        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in tqdm(range(epoch)):
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs = dataset.next_batch(batch_size, batch_keys=['Xs'])
                noise = self.get_noise(Xs.shape)
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._noise: noise})

            Xs = dataset.next_batch(batch_size, batch_keys=['Xs'], look_up=False)
            noise = self.get_noise(Xs.shape)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._noise: noise})
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=np.mean(loss)))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        noise = self.get_noise(Xs.shape)
        return self.sess.run(self._code_ops, feed_dict={self._Xs: Xs, self._noise: noise})

    def recon(self, Xs):
        noise = self.get_noise(Xs.shape)
        return self.sess.run(self._recon_ops, feed_dict={self._Xs: Xs, self._noise: noise})

    def generate(self, zs):
        return self.sess.run(self._recon_ops, feed_dict={self._zs: zs})

    def metric(self, Xs):
        noise = self.get_noise(Xs.shape)
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._noise: noise})
