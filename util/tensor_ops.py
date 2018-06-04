"""operation util for tensorflow"""

import tensorflow as tf

"""convolution filter option
(kernel height, kernel width, stride height, stride width)
"""
CONV_FILTER_1111 = (1, 1, 1, 1)
CONV_FILTER_2211 = (2, 2, 1, 1)
CONV_FILTER_2222 = (2, 2, 2, 2)
CONV_FILTER_3311 = (3, 3, 1, 1)
CONV_FILTER_3322 = (3, 3, 2, 2)
CONV_FILTER_4411 = (4, 4, 1, 1)
CONV_FILTER_4422 = (4, 4, 2, 2)
CONV_FILTER_5511 = (5, 5, 1, 1)
CONV_FILTER_5522 = (5, 5, 2, 2)
CONV_FILTER_5533 = (5, 5, 3, 3)
CONV_FILTER_6611 = (6, 6, 1, 1)
CONV_FILTER_6622 = (6, 6, 2, 2)
CONV_FILTER_7711 = (7, 7, 1, 1)
CONV_FILTER_7722 = (7, 7, 2, 2)
CONV_FILTER_9911 = (9, 9, 1, 1)
CONV_FILTER_9922 = (9, 9, 2, 2)


# normalization
def bn(x, is_training=True, name='bn'):
    """batch normalization layer"""
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


# activation function
def sigmoid(x, name='sigmoid'):
    """sigmoid activation function layer"""
    return tf.sigmoid(x, name=name)


def tanh(x, name='tanh'):
    """tanh activation function layer"""
    return tf.tanh(x, name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    """leak relu activate function layer"""
    return tf.maximum(x, leak * x, name=name)


def relu(input_, name='relu'):
    """relu activate function layer"""
    return tf.nn.relu(features=input_, name=name)


def elu(input_, name="elu"):
    """elu activate function layer"""
    return tf.nn.elu(features=input_, name=name)


def linear(input_, output_size, name="linear", stddev=0.02, bias_start=0.0, with_w=False):
    """pre-activated linear layer

    typical one layer of neural net, return just before activate
    input * weight + bias

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_size: int
    :type name: str
    :type stddev: float
    :type bias_start: float
    :type with_w: bool
    :param input_: input variable or placeholder of tensorflow
    :param output_size: output layer size
    :param name: tensor scope name
    :param stddev: stddev for initialize weight
    :param bias_start: initial value of baise
    :param with_w: return with weight and bias tensor variable

    :return: before activate neural net
    :rtype tensorflow.Variable

    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, weight) + bias, weight, bias
        else:
            return tf.matmul(input_, weight) + bias


def conv2d_transpose(input_, output_shape, filter_, name="conv2d_transpose", stddev=0.02,
                     with_w=False):
    """transposed 2d convolution layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_shape: Union[list, tuple]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type with_w: bool
    :type stddev: float
    :param input_: input variable or placeholder of tensorflow
    :param output_shape: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param stddev: stddev for initialize weight
    :param with_w: return with weight and bias tensor variable

    :return: result of 2d transposed convolution
    :rtype tensorflow.Variable
    """
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
    """2d convolution layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type stddev: float
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param stddev: stddev for initialize weight

    :return: result of 2d convolution
    :rtype tensorflow.Variable
    """
    k_h, k_w, d_h, d_w = filter_
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, weight, strides=[1, d_h, d_w, 1], padding='SAME')

        bias = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        return conv


def conv2d_one_by_one(input_, output_channel, name='conv2d_one_by_one'):
    """bottle neck convolution layer

    1*1 kernel 1*1 stride convolution

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param name: tensor scope name

    :return: bottle neck convolution
    :rtype tensorflow.Variable
    """
    out = conv2d(input_, output_channel, CONV_FILTER_1111, name=name)
    return out


def upscale_2x(input_, output_channel, filter_, name='upscale_2x'):
    """transposed convolution to double scale up layer

    doubled width and height of input

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name

    :return: result of 2*2 upscale
    :rtype tensorflow.Variable
    """
    shape = input_.get_shape()
    n, h, w, _ = shape
    output_shape = [int(n), int(h) * 2, int(w) * 2, int(output_channel)]
    return conv2d_transpose(input_, output_shape, filter_, name=name)


def upscale_2x_block(input_, output_channel, filter_, activate, name='upscale_2x_block'):
    """2*2 upscale tensor block(transposed convolution, batch normalization, activation)

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type activate: func
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after transposed convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param activate: activate function

    :return: result of tensor block
    :rtype tensorflow.Variable
    """
    with tf.variable_scope(name):
        input_ = upscale_2x(input_, output_channel, filter_)
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def conv_block(input_, output_channel, filter_, activate, name='conv_block'):
    """convolution tensor block(convolution, batch normalization, activation)

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type output_channel: int
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :type activate: func
    :param input_: input variable or placeholder of tensorflow
    :param output_channel: output shape of after convolution
    :param filter_: convolution filter(kernel and stride)
    :param name: tensor scope name
    :param activate: activate function

    :return: result of tensor block
    :rtype tensorflow.Variable
    """
    with tf.variable_scope(name):
        input_ = conv2d(input_, output_channel, filter_, name='conv')
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def linear_block(input_, output_size, activate, name="linear_block"):
    with tf.variable_scope(name):
        input_ = linear(input_, output_size)
        input_ = bn(input_)
        input_ = activate(input_)
    return input_


def avg_pooling(input_, filter_, name='avg_pooling'):
    """average pooling layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param filter_: pooling filter(kernel and stride)
    :param name: tensor scope name

    :return: result of average pooling
    :rtype tensorflow.Variable
    """
    kH, kW, sH, sW = filter_
    return tf.nn.avg_pool(input_, ksize=[1, kH, kW, 1], strides=[1, sH, sW, 1], padding='SAME', name=name)


def max_pooling(input_, filter_, name='max_pooling'):
    """max pooling layer

    :type input_: Union[tensorflow.Variable, tensorflow.PlaceHolder]
    :type filter_: tuple[int, int, int, int]
    :type name: str
    :param input_: input variable or placeholder of tensorflow
    :param filter_: pooling filter(kernel and stride)
    :param name: tensor scope name

    :return: result of max pooling
    :rtype tensorflow.Variable
    """
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
    """softmax layer"""
    return tf.nn.softmax(input_, name=name)


def dropout(input_, rate, name="dropout"):
    """dropout"""
    return tf.nn.dropout(input_, rate, name=name)


def L1_norm(var_list, lambda_=1.0, name="L1_norm"):
    return tf.multiply(lambda_,
                       tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in var_list]),
                       name=name)


def L2_norm(var_list, lambda_=1.0, name="L2_norm"):
    return tf.multiply(lambda_,
                       tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(tf.abs(var))) for var in var_list])),
                       name=name)


def wall_decay(decay_rate, global_step, wall_step, name='decay'):
    return tf.pow(decay_rate, global_step // wall_step, name=name)


def average_top_k_loss(loss, k, name='average_top_k_loss'):
    values, indices = tf.nn.top_k(loss, k=k, name=name)
    return values


def reshape(input_, shape, name='reshape'):
    if shape[0] == None:
        shape[0] = -1
    return tf.reshape(input_, shape, name=name)


def concat(values, axis, name="concat"):
    return tf.concat(values, axis, name=name)


def flatten(input_, name='flatten'):
    return tf.layers.flatten(input_, name=name)


def join_scope(*args, spliter='/'):
    return spliter.join(map(str, args))


def split_scope(scope, spliter='/'):
    return str(scope).split(spliter)


def get_scope():
    return tf.get_variable_scope().name


def collect_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def placeholder(dtype, shape, name):
    if shape[0] == -1:
        shape[0] = None
    return tf.placeholder(dtype=dtype, shape=shape, name=name)
