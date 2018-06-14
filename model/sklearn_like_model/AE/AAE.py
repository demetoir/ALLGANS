from model.sklearn_like_model.BaseModel import BaseModel
from model.sklearn_like_model.DummyDataset import DummyDataset
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
from functools import reduce
import numpy as np


class AAE(BaseModel):

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
            'D_gauss_net_shapes',
            ''
        ]

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _Ys(self):
        return self.Ys

    @property
    def _zs(self):
        return self.zs

    @property
    def _train_ops(self):
        return [self.train_AE, self.train_D_gauss, self.train_D_cate, self.train_G_gauss, self.train_G_cate,
                self.train_clf, self.op_inc_global_step]

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
        return [self.loss_AE,
                self.loss_G_gauss,
                self.loss_G_cate,
                self.loss_D_gauss,
                self.loss_D_cate,
                self.loss_clf]

    @property
    def _predict_ops(self):
        return self.predict_index

    @property
    def _score_ops(self):
        return self.acc_mean

    @property
    def _proba_ops(self):
        return self.hs

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
        self.D_gauss_net_shapes = [512, 512]
        self.D_cate_net_shapes = [512, 512]

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape
        self.X_flatten_size = reduce(lambda x, y: x * y, self.X_shape)

        self.Y_batch_key = 'Ys'
        self.Y_shape = input_shapes[self.Y_batch_key]
        self.Ys_shape = [None] + self.Y_shape
        self.Y_size = self.Y_shape[0]

        self.z_shape = [self.z_size]
        self.zs_shape = [None] + self.z_shape

    def encoder(self, Xs, net_shapes, reuse=False, name='encoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(Xs)
            stack.flatten()

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.z_size + self.Y_size, relu)
            zs = stack.last_layer[:, :self.z_size]
            Ys_gen = stack.last_layer[:, self.z_size:]

            hs = softmax(Ys_gen)
        return zs, Ys_gen, hs

    def decoder(self, zs, Ys, net_shapes, reuse=False, name='decoder'):
        with tf.variable_scope(name, reuse=reuse):
            stack = Stacker(concat((zs, Ys), axis=1))

            for shape in net_shapes:
                stack.linear_block(shape, relu)

            stack.linear_block(self.X_flatten_size, sigmoid)
            stack.reshape(self.Xs_shape)

        return stack.last_layer

    def discriminator_gauss(self, zs, net_shapes, reuse=False, name='discriminator_gauss'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
            for shape in net_shapes:
                layer.linear_block(shape, relu)

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def discriminator_cate(self, zs, net_shapes, reuse=False, name='discriminator_cate'):
        with tf.variable_scope(name, reuse=reuse):
            layer = Stacker(zs)
            for shape in net_shapes:
                layer.linear_block(shape, relu)

            layer.linear(1)
            layer.sigmoid()

        return layer.last_layer

    def build_main_graph(self):
        self.Xs = placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.zs = placeholder(tf.float32, self.zs_shape, name='zs')

        self.zs_gen, self.Ys_gen, self.hs = self.encoder(self.Xs, self.encoder_net_shapes)
        self.latent_code = self.zs_gen
        self.Xs_recon = self.decoder(self.zs_gen, self.Ys_gen, self.decoder_net_shapes)
        self.Xs_gen = self.decoder(self.zs, self.Ys, self.decoder_net_shapes, reuse=True)

        self.D_gauss_real = self.discriminator_gauss(self.zs, self.D_gauss_net_shapes)
        self.D_gauss_gen = self.discriminator_gauss(self.zs_gen, self.D_gauss_net_shapes, reuse=True)

        self.D_cate_real = self.discriminator_cate(self.Ys, self.D_cate_net_shapes)
        self.D_cate_gen = self.discriminator_cate(self.Ys_gen, self.D_cate_net_shapes, reuse=True)

        head = get_scope()
        self.vars_encoder = collect_vars(join_scope(head, 'encoder'))
        self.vars_decoder = collect_vars(join_scope(head, 'decoder'))
        self.vars_discriminator_gauss = collect_vars(join_scope(head, 'discriminator_gauss'))
        self.vars_discriminator_cate = collect_vars(join_scope(head, 'discriminator_cate'))

        # self.predict = tf.equal(tf.argmax(self.hs, 1), tf.argmax(self.Ys, 1), name='predict')

        self.predict_index = tf.cast(tf.argmax(self.hs, 1), tf.float32, name="predict_index")
        self.label_index = onehot_to_index(self.Ys)
        self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def build_loss_function(self):
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

    def build_train_ops(self):
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

    def random_z(self):
        pass

    def get_z_noise(self, shape):
        return np.random.uniform(-1, 1, size=shape)

    def train(self, Xs, Ys, epoch=100, save_interval=None, batch_size=None):
        self.if_not_ready_to_train()

        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in range(epoch):
            dataset.shuffle()
            total_AE = 0
            total_G_gauss = 0
            total_G_cate = 0
            total_D_gauss = 0
            total_D_cate = 0
            total_clf = 0
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                # print([batch_size, self.z_size])
                zs = self.get_z_noise([batch_size, self.z_size])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs})
                loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs})

                loss_AE, loss_G_gauss, loss_G_cate, loss_D_gauss, loss_D_cate, loss_clf = loss
                total_AE += np.sum(loss_AE) / dataset.size
                total_G_gauss += np.sum(loss_G_gauss) / dataset.size
                total_G_cate += np.sum(loss_G_cate) / dataset.size
                total_D_gauss += np.sum(loss_D_gauss) / dataset.size
                total_D_cate += np.sum(loss_D_cate) / dataset.size
                total_clf += np.sum(loss_clf) / dataset.size

            self.log.info(
                "e:{} loss AE={}, G_gauss={}, G_cate={}, D_gauss={}, D_cate={}, "
                "clf={}".format(e, total_AE, total_G_gauss, total_G_cate, total_D_gauss, total_D_cate, total_clf))
            # Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], look_up=False)
            # zs = self.get_z_noise([batch_size, self.z_size])
            # loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs})

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        return self.sess.run(self._code_ops, feed_dict={self._Xs: Xs})

    def recon(self, Xs, Ys):
        return self.sess.run(self._recon_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

    def generate(self, zs, Ys):
        return self.sess.run(self._recon_ops, feed_dict={self._zs: zs, self._Ys: Ys})

    def metric(self, Xs, Ys):
        Xs = np.array(Xs)
        zs = self.get_z_noise([Xs.shape[0], self.z_size])
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys, self._zs: zs})

    def proba(self, Xs):
        return self.sess.run(self._proba_ops, feed_dict={self._Xs: Xs})

    def predict(self, Xs):
        return self.sess.run(self._predict_ops, feed_dict={self._Xs: Xs})

    def score(self, Xs, Ys):
        return self.sess.run(self._score_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
