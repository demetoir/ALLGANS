from util.tensor_ops import *


class Stacker:
    """help easily make graph model by stacking layer and naming for tensorflow

    ex)
    stacker = Stacker(input_)
    stacker.add_layer(conv2d, 64, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)

    stacker.add_layer(conv2d, 128, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)

    stacker.add_layer(conv2d, 256, CONV_FILTER_5522)
    stacker.add_layer(bn)
    stacker.add_layer(lrelu)
    last_layer = stacker.last_layer
    """

    def __init__(self, start_layer=None, reuse=False, name="stacker"):
        """create SequenceModel

        :param start_layer:the start layer
        :param reuse:reuse option for tensorflow graph
        :param name:prefix name for layer
        """
        self.reuse = reuse
        self.layer_count = 1
        self.last_layer = start_layer
        self.layer_seq = [start_layer]
        self.name = name

    def add_layer(self, func, *args, **kwargs):
        """add new layer right after last added layer

        :param func: function for tensor layer
        :param args: args for layer
        :param kwargs: kwargs for layer
        :return: added new layer
        """
        scope_name = self.name + '_layer' + str(self.layer_count)
        with tf.variable_scope(scope_name, reuse=self.reuse):
            if func == concat:
                self.last_layer = func(*args, **kwargs)
            else:
                self.last_layer = func(self.last_layer, *args, **kwargs)

            self.layer_seq += [self.last_layer]
            self.layer_count += 1
        return self.last_layer

    def bn(self):
        """add batch normalization layer"""
        return self.add_layer(bn)

    def sigmoid(self):
        """add sigmoid layer"""
        return self.add_layer(sigmoid)

    def tanh(self):
        return self.add_layer(tanh)

    def lrelu(self):
        """add leaky relu layer"""
        return self.add_layer(lrelu)

    def relu(self):
        """add relu layer"""
        return self.add_layer(relu)

    def elu(self):
        """add elu layer"""
        return self.add_layer(elu)

    def linear(self, output_size):
        """add linear layer"""
        return self.add_layer(linear, output_size)

    def linear_block(self, output_size, activate):
        return self.add_layer(linear_block, output_size, activate)

    def conv2d_transpose(self, output_shape, filter_):
        """add 2d transposed convolution layer"""
        return self.add_layer(conv2d_transpose, output_shape, filter_)

    def conv2d(self, output_channel, filter_):
        """add 2d convolution layer"""
        return self.add_layer(conv2d, output_channel, filter_)

    def conv2d_one_by_one(self, output_channel):
        """add bottle neck convolution layer"""
        return self.add_layer(conv2d_one_by_one, output_channel)

    def upscale_2x(self, output_channel, filter_):
        """add upscale 2x layer"""
        return self.add_layer(upscale_2x, output_channel, filter_)

    def upscale_2x_block(self, output_channel, filter_, activate):
        """add upscale 2x block layer"""
        return self.add_layer(upscale_2x_block, output_channel, filter_, activate)

    def conv_block(self, output_channel, filter_, activate):
        """add convolution block layer"""
        return self.add_layer(conv_block, output_channel, filter_, activate)

    def avg_pooling(self, filter_):
        """add average pooling layer"""
        return self.add_layer(avg_pooling, filter_)

    def max_pooling(self, filter_):
        """add max pooling layer"""
        return self.add_layer(max_pooling, filter_)

    def softmax(self):
        """add softmax layer"""
        return self.add_layer(softmax)

    def dropout(self, rate):
        """add dropout layer"""
        return self.add_layer(dropout, rate)

    def reshape(self, shape):
        return self.add_layer(reshape, shape)

    def concat(self, values, axis):
        return self.add_layer(concat, values, axis)

    def flatten(self):
        return self.add_layer(flatten)
