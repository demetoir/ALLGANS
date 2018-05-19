from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *


class MLPClassifier(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_hyper_parameter(self):
        self.batch_size = 100
        self.learning_rate = 0.01
        self.beta1 = 0.5
        # self.DROPOUT_RATE = 0.5
        self.K_average_top_k_loss = 20

    def load_input_shapes(self, input_shapes):
        self.X_batch_key = 'Xs'
        self.X_shape = input_shapes[self.X_batch_key]
        self.Xs_shape = [self.batch_size] + self.X_shape

        self.Y_batch_key = 'Ys'
        self.Y_shape = input_shapes[self.Y_batch_key]
        self.Ys_shape = [self.batch_size] + self.Y_shape
        self.Y_size = self.Y_shape[0]

    def classifier(self, x, dropout_rate):
        with tf.variable_scope('classifier'):
            layer = Stacker(x)

            layer.linear_block(128, lrelu)
            layer.dropout(dropout_rate)

            layer.linear_block(128, relu)
            layer.dropout(dropout_rate)

            layer.linear(2)
            logit = layer.last_layer

            h = softmax(logit)
        return logit, h

    def load_main_tensor_graph(self):
        self.Xs = tf.placeholder(tf.float32, self.Xs_shape, name='Xs')
        self.Ys = tf.placeholder(tf.float32, self.Ys_shape, name='Ys')
        self.dropout_rate = tf.placeholder(tf.float32, [], name='dropout_rate')

        self.logit, self.h = self.classifier(self.Xs, self.dropout_rate)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.Ys)
        self.acc = tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64, name="acc")
        self.acc_mean = tf.reduce_mean(self.acc, name="acc_mean")

    def load_loss_function(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Ys, logits=self.logit)

        self.l1_norm_penalty = L1_norm(self.vars, lambda_=0.0001)
        self.l1_norm_penalty_mean = tf.reduce_mean(self.l1_norm_penalty, name='l1_norm_penalty_mean')
        # self.l1_norm_penalty *= wall_decay(0.999, self.global_step, 100)
        self.l2_norm_penalty = L2_norm(self.vars, lambda_=0.2)
        self.l2_norm_penalty_mean = tf.reduce_mean(self.l2_norm_penalty, name='l2_norm_penalty_mean')

        self.loss = self.loss + self.l1_norm_penalty
        # average top k loss
        # self.loss = average_top_k_loss(self.loss, self.K_average_top_k_loss)
        self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

    def load_train_ops(self):
        with tf.variable_scope('train_ops'):
            self.train = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss, var_list=self.vars)

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs, batch_labels = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key]
        )
        sess.run(
            [self.train, self.op_inc_global_step],
            feed_dict={
                self.Xs: batch_xs,
                self.Ys: batch_labels,
                self.dropout_rate: 1
            }
        )

    def run_model(self, sess=None, ops=None, feed_dict=None):
        feed_dict[self.dropout_rate] = 1
        return sess.run(ops, feed_dict)

    def load_summary_ops(self):
        summary_loss(self.loss)

        self.op_merge_summary = tf.summary.merge_all()

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        batch_xs, batch_labels = dataset.train_set.next_batch(
            self.batch_size,
            batch_keys=[self.X_batch_key, self.Y_batch_key]
        )
        summary, global_step = sess.run(
            [self.op_merge_summary, self.global_step],
            feed_dict={
                self.Xs: batch_xs,
                self.Ys: batch_labels,
                self.dropout_rate: 0.6
            }
        )
        summary_writer.add_summary(summary, global_step)

    def run(self, sess, fetches, dataset):
        Xs, Ys = dataset.train_set.next_batch(
            self.batch_size,
            batch_size=[self.X_batch_key, self.Y_batch_key]
        )
        return self.get_tf_values(sess, fetches, Xs, Ys)

    def get_tf_values(self, sess, fetches, Xs, Ys):
        return sess.run(
            fetches,
            feed_dict={
                self.Xs: Xs,
                self.Ys: Ys,
                self.dropout_rate: 1
            }
        )
