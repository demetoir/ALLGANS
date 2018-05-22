from model.AbstractModel import AbstractModel
from util.Stacker import Stacker
from util.tensor_ops import *
from util.summary_func import *
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *


def inception_layer(input_, channel_size, name='inception_layer'):
    with tf.variable_scope(name):
        with tf.variable_scope('out1'):
            layer = Stacker(input_)
            layer.add_layer(avg_pooling, CONV_FILTER_2211)
            out1 = layer.last_layer

        with tf.variable_scope('out2'):
            layer = Stacker(input_)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, lrelu)
            out2 = layer.last_layer

        with tf.variable_scope('out3'):
            layer = Stacker(input_)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, lrelu)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, relu)
            out3 = layer.last_layer

        with tf.variable_scope('out4'):
            layer = Stacker(input_)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, lrelu)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, lrelu)
            layer.add_layer(conv_block, channel_size, CONV_FILTER_5511, lrelu)
            out4 = layer.last_layer

        out = tf.concat([out1, out2 + out3 + out4], 3)

        return out


class Classifier(AbstractModel):
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __str__(self):
        return "Classifier"

    def load_hyper_parameter(self, params=None):
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.disc_iters = 1

    def load_input_shapes(self, input_shapes):
        shape_data_x = input_shapes[INPUT_SHAPE_KEY_DATA_X]
        if len(shape_data_x) == 3:
            self.shape_data_x = shape_data_x
            H, W, C = shape_data_x
            self.input_size = W * H * C
            self.input_w = W
            self.input_h = H
            self.input_c = C
        elif len(shape_data_x) == 2:
            self.shape_data_x = shape_data_x + [1]
            H, W = shape_data_x
            self.input_size = W * H
            self.input_w = W
            self.input_h = H
            self.input_c = 1

        self.label_shape = input_shapes[INPUT_SHAPE_KEY_LABEL]
        self.label_size = input_shapes[INPUT_SHAPE_KEY_LABEL_SIZE]

    def CNN(self, input_):
        with tf.variable_scope('classifier'):
            layer = Stacker(input_, name='seq1')
            layer.add_layer(conv_block, 64, CONV_FILTER_5522, lrelu)
            size16 = layer.last_layer
            layer.add_layer(inception_layer, 32)
            layer.add_layer(inception_layer, 64)
            layer.add_layer(inception_layer, 128)
            layer.add_layer(tf.reshape, [self.batch_size, -1])

            layer2 = Stacker(size16, name='seq2')
            layer2.add_layer(conv_block, 128, CONV_FILTER_5522, lrelu)
            size8 = layer2.last_layer
            layer2.add_layer(inception_layer, 64)
            layer2.add_layer(inception_layer, 128)
            layer2.add_layer(inception_layer, 256)
            layer2.add_layer(tf.reshape, [self.batch_size, -1])

            layer3 = Stacker(size8, name='seq3')
            layer3.add_layer(conv_block, 256, CONV_FILTER_5522, lrelu)
            layer3.add_layer(inception_layer, 128)
            layer3.add_layer(inception_layer, 256)
            layer3.add_layer(inception_layer, 512)
            layer3.add_layer(tf.reshape, [self.batch_size, -1])

            merge = tf.concat([layer.last_layer, layer2.last_layer, layer3.last_layer], axis=1)
            after_merge = Stacker(merge, name='after_merge')
            after_merge.add_layer(linear, self.label_size)

            logit = after_merge.last_layer
            h = softmax(logit)

        return logit, h

    def load_main_tensor_graph(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size] + self.shape_data_x, name='X')
        self.label = tf.placeholder(tf.float32, [self.batch_size] + self.label_shape, name='label')

        self.logit, self.h = self.CNN(self.X)

        self.predict_index = tf.cast(tf.argmax(self.h, 1, name="predicted_label"), tf.float32)
        self.label_index = onehot_to_index(self.label)
        self.batch_acc = tf.reduce_mean(tf.cast(tf.equal(self.predict_index, self.label_index), tf.float64),
                                        name="batch_acc")

    def load_loss_function(self):
        with tf.variable_scope('loss'):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logit)
            self.loss_mean = tf.reduce_mean(self.loss)

    def load_train_ops(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')

        with tf.variable_scope('train_ops'):
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1) \
                .minimize(self.loss, var_list=self.vars)

        with tf.variable_scope('clip_op'):
            self.clip_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.vars]

    def train_model(self, sess=None, iter_num=None, dataset=None):
        batch_xs, batch_labels = dataset.next_batch(self.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        sess.run([self.train, self.clip_op], feed_dict={self.X: batch_xs, self.label: batch_labels})

        sess.run([self.op_inc_global_step])

    def load_summary_ops(self):
        summary_loss(self.loss)

        self.op_merge_summary = tf.summary.merge_all()

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        batch_xs, batch_labels = dataset.next_batch(self.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        summary, global_step = sess.run([self.op_merge_summary, self.global_step],
                                        feed_dict={self.X: batch_xs, self.label: batch_labels})
        summary_writer.add_summary(summary, global_step)
