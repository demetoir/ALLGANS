from model.sklearn_like_model.BaseClassifierModel import BaseClassifierModel
from util.Stacker import Stacker
from util.tensor_ops import *
import tensorflow as tf


class MLPClassifier(BaseClassifierModel):
    VERSION = 1.0

    @property
    def hyper_param_key(self):
        return [
            'batch_size',
            'learning_rate',
            'beta1',
            'dropout_rate',
            'K_average_top_k_loss',
            'net_shapes',
            'activation',
            'l1_norm_lambda',
            'l2_norm_lambda'
        ]

    @property
    def _Xs(self):
        return self.Xs

    @property
    def _Ys(self):
        return self.Ys

    @property
    def _predict_ops(self):
        return self.predict_index

    @property
    def _score_ops(self):
        return self.acc_mean

    @property
    def _proba_ops(self):
        return self.h

    @property
    def _metric_ops(self):
        return self.loss

    @property
    def _train_ops(self):
        return [self.train_op, self.op_inc_global_step]

    def hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        self.K_average_top_k_loss = 20
        self.net_shapes = [128, 128]
        self.activation = 'relu'
        self.l1_norm_lambda = 0.0001
        self.l2_norm_lambda = 0.001

    def build_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [None] + self.X_shape

        self.Y_batch_key = 'Ys'
        self.Y_shape = input_shapes[self.Y_batch_key]
        self.Ys_shape = [None] + self.Y_shape
        self.Y_size = self.Y_shape[0]

    def classifier(self, Xs, net_shapes, name='classifier'):
        with tf.variable_scope(name):
            layer = Stacker(flatten(Xs))

            for net_shape in net_shapes:
                layer.linear_block(net_shape, relu)

            layer.linear(self.Y_size)
            logit = layer.last_layer
            h = softmax(logit)
        return logit, h

    def build_main_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')

        self.logit, self.h = self.classifier(self.Xs, self.net_shapes)

        self.vars = collect_vars(join_scope(get_scope(), 'classifier'))

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.Ys)
        self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def build_loss_function(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys, logits=self.logit)

        self.l1_norm_penalty = L1_norm(self.vars, lambda_=self.l1_norm_lambda)
        self.l1_norm_penalty_mean = tf.reduce_mean(self.l1_norm_penalty, name='l1_norm_penalty_mean')
        # self.l1_norm_penalty *= wall_decay(0.999, self.global_step, 100)
        self.l2_norm_penalty = L2_norm(self.vars, lambda_=self.l2_norm_lambda)
        self.l2_norm_penalty_mean = tf.reduce_mean(self.l2_norm_penalty, name='l2_norm_penalty_mean')

        self.loss = self.loss + self.l1_norm_penalty
        # average top k loss
        # self.loss = average_top_k_loss(self.loss, self.K_average_top_k_loss)
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def build_train_ops(self):
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars)
