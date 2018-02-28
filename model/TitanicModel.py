from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
from data_handler.titanic import *


class TitanicModel(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def load_hyper_parameter(self):
        self.batch_size = 50
        self.learning_rate = 0.0002
        self.beta1 = 0.5

    def load_input_shapes(self, input_shapes):
        self.x_shape = input_shapes[ISK_TRAIN_X]
        self.label_shape = input_shapes[ISK_TRAIN_LABEL]

    def classifier(self, x):
        with tf.variable_scope('classifier'):
            layer = Stacker(x)
            layer.linear(512)
            layer.bn()
            layer.lrelu()

            layer.linear(512)
            layer.bn()
            layer.lrelu()

            layer.linear(512)
            layer.bn()
            layer.lrelu()

            logit = layer.linear(2)

            h = softmax(logit)
        return logit, h

    def load_main_tensor_graph(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size] + self.x_shape, name='X')
        self.label = tf.placeholder(tf.float32, [self.batch_size] + self.label_shape, name='label')

        self.logit, self.h = self.classifier(self.X)

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.label)
        self.batch_acc = tf.reduce_mean(tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64),
                                        name="batch_acc")

    def load_loss_function(self):
        with tf.variable_scope('loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logit)
            self.loss_mean = tf.reduce_mean(self.loss)

    def load_train_ops(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')

        with tf.variable_scope('train_ops'):
            self.train = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=self.beta1) \
                .minimize(self.loss, var_list=self.vars)

        # with tf.variable_scope('clip_op'):
        #     self.clip_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.vars]

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs, batch_labels = dataset.next_batch(self.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        # sess.run([self.train, self.clip_op], feed_dict={self.X: batch_xs, self.label: batch_labels})
        sess.run([self.train], feed_dict={self.X: batch_xs, self.label: batch_labels})

        sess.run([self.op_inc_global_step])

    def load_summary_ops(self):
        summary_loss(self.loss)

        self.op_merge_summary = tf.summary.merge_all()

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        batch_xs, batch_labels = dataset.next_batch(
            self.batch_size,
            batch_keys=[
                BATCH_KEY_TRAIN_X,
                BATCH_KEY_TRAIN_LABEL
            ]
        )
        summary, global_step = sess.run([self.op_merge_summary, self.global_step],
                                        feed_dict={self.X: batch_xs, self.label: batch_labels})
        summary_writer.add_summary(summary, global_step)
