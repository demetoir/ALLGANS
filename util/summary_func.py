"""tensorflow summary util"""
import tensorflow as tf


def mean_summary(var):
    """mean scalar summary

    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def stddev_summary(var):
    """stddev scalar summary

    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)


def histogram_summary(var):
    """histogram summary

    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.histogram('histogram', var)


def max_summary(var):
    """max scalar summary

    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.scalar("max", tf.reduce_max(var))


def min_summary(var):
    """min summary

    :type var: tensorflow.Variable
    :param var: variable to add summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.scalar("min", tf.reduce_min(var))


def summary_loss(var):
    """loss summary

    loss's scalar and histogram summary

    :type var: tensorflow.Variable
    :param var: variable to summary
    """
    with tf.name_scope(var.name.split(":")[0]):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


def summary_image(var, max_outputs=0):
    """image summary

    :type var: tensorflow.Variable
    :type max_outputs: int
    :param var: variable to summary
    :param max_outputs: max output to summary image
    """
    with tf.name_scope(var.name.split(":")[0]):
        tf.summary.image("image", var, max_outputs=max_outputs)
