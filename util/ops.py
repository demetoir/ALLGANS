import tensorflow as tf

filter_1111 = (1, 1, 1, 1)

filter_2211 = (2, 2, 1, 1)
filter_2222 = (2, 2, 2, 2)

filter_3311 = (3, 3, 1, 1)
filter_3322 = (3, 3, 2, 2)

filter_4411 = (4, 4, 1, 1)
filter_4422 = (4, 4, 2, 2)

filter_5511 = (5, 5, 1, 1)
filter_5522 = (5, 5, 2, 2)
filter_5533 = (5, 5, 3, 3)

filter_6611 = (6, 6, 1, 1)
filter_6622 = (6, 6, 2, 2)

filter_7711 = (7, 7, 1, 1)
filter_7722 = (7, 7, 2, 2)

filter_9911 = (9, 9, 1, 1)
filter_9922 = (9, 9, 2, 2)


def bn(x, is_training=True, name='bn'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


# activation function
def sigmoid(x, name='sigmoid'):
    return tf.sigmoid(x, name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name)


def relu(input_, name='relu'):
    return tf.nn.relu(features=input_, name=name)


def elu(input_, name="elu"):
    return tf.nn.elu(features=input_, name=name)


def linear(input_, output_size, name="linear", stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, weight) + bias, weight, bias
        else:
            return tf.matmul(input_, weight) + bias


def conv2d_transpose(input_, output_shape, filter_, name="conv2d_transpose", stddev=0.02,
                     with_w=False):
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                 initializer=tf.random_normal_initializer(stddev=stddev))

        conv_transpose = tf.nn.conv2d_transpose(input_, weight, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        conv_transpose = tf.reshape(tf.nn.bias_add(conv_transpose, bias), conv_transpose.get_shape())

        if with_w:
            return conv_transpose, weight, bias
        else:
            return conv_transpose


def conv2d(input_, output_channel, filter_, stddev=0.02, name="conv2d"):
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weight, strides=[1, d_h, d_w, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        return conv


def conv2d_one_by_one(input_, output_channel, name='conv2d_one_by_one'):
    out = conv2d(input_, output_channel, filter_1111, name=name)
    return out


def upscale_2x(input_, output_channel, filter_, name='upscale_2x'):
    shape = input_.get_shape()
    n, h, w, _ = shape
    output_shape = [int(n), int(h) * 2, int(w) * 2, int(output_channel)]
    return conv2d_transpose(input_, output_shape, filter_, name=name)


def upscale_2x_block(net, output_channel, filter_, activate, name='upscale_2x_block'):
    with tf.variable_scope(name):
        net = upscale_2x(net, output_channel, filter_)
        net = bn(net)
        net = activate(net)
    return net


def conv_block(net, output_channel, filter_, activate, name='conv_block'):
    with tf.variable_scope(name):
        net = conv2d(net, output_channel, filter_, name='conv')
        net = bn(net, name='bn')
        net = activate(net)
    return net


def avg_pooling(input_, filter_, name='avg_pooling'):
    kH, kW, sH, sW = filter_
    return tf.nn.avg_pool(input_, ksize=[1, kH, kW, 1], strides=[1, sH, sW, 1], padding='SAME', name=name)


def max_pooling(input_, filter_, name='max_pooling'):
    kH, kW, sH, sW = filter_
    return tf.nn.max_pool(input_, ksize=[1, kH, kW, 1], strides=[1, sH, sW, 1], padding='SAME', name=name)


def onehot_to_index(onehot):
    with tf.variable_scope('onehot_to_index'):
        index = tf.cast(tf.argmax(onehot, 1), tf.float32)
    return index


def index_to_onehot(index, size):
    # TODO implement
    onehot = tf.one_hot(index, size)
    return onehot


def softmax(input_, name='softmax'):
    return tf.nn.softmax(input_, name=name)
