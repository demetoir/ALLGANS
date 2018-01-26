from util.ops import *
from util.SequenceModel import SequenceModel


class stable_model_1:
    """
    GAN tested
    WGAN
    LSGAN
    """

    @staticmethod
    def generator(z, batch_size):
        seq = SequenceModel(z)
        seq.add_layer(linear, 4 * 4 * 512)
        seq.add_layer(tf.reshape, [batch_size, 4, 4, 512])

        seq.add_layer(conv2d_transpose, [batch_size, 8, 8, 256], filter_5522)
        seq.add_layer(bn)
        seq.add_layer(relu)

        seq.add_layer(conv2d_transpose, [batch_size, 16, 16, 128], filter_5522)
        seq.add_layer(bn)
        seq.add_layer(relu)

        seq.add_layer(conv2d_transpose, [batch_size, 32, 32, 3], filter_5522)
        seq.add_layer(conv2d, 3, filter_3311)
        seq.add_layer(tf.sigmoid)
        net = seq.last_layer

        return net

    @staticmethod
    def discriminator(x, batch_size):
        seq = SequenceModel(x)
        seq.add_layer(conv2d, 64, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 128, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 256, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 256, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(tf.reshape, [batch_size, -1])
        out_logit = seq.add_layer(linear, 1)
        out = seq.add_layer(tf.sigmoid)

        return out, out_logit


class stable_model_2:
    @staticmethod
    def test_generator(z, batch_size):
        seq = SequenceModel(z)
        seq.add_layer(linear, 4 * 4 * 512)
        seq.add_layer(tf.reshape, [batch_size, 4, 4, 512])

        seq.add_layer(conv2d_transpose, [batch_size, 8, 8, 256], filter_7722)
        seq.add_layer(bn)
        seq.add_layer(relu)

        seq.add_layer(conv2d_transpose, [batch_size, 16, 16, 128], filter_7722)
        seq.add_layer(bn)
        seq.add_layer(relu)

        seq.add_layer(conv2d_transpose, [batch_size, 32, 32, 64], filter_7722)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 3, filter_5511)
        seq.add_layer(tf.sigmoid)

        return seq.last_layer

    @staticmethod
    def discriminator(x, batch_size):
        seq = SequenceModel(x)
        seq.add_layer(conv2d, 64, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 128, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 256, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(conv2d, 256, filter_5522)
        seq.add_layer(bn)
        seq.add_layer(lrelu)

        seq.add_layer(tf.reshape, [batch_size, -1])
        out_logit = seq.add_layer(linear, 1)
        out = seq.add_layer(tf.sigmoid)

        return out, out_logit
