import tensorflow as tf


def summary_variable(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    scope_str = str(var.name)[:-2]
    with tf.variable_scope(scope_str, reuse=True):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.variable_scope('stddev', reuse=True):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def summary_variable_mean(var, var_name='variable_mean'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    scope_str = str(var.name)[:-2]

    mean = tf.reduce_mean(var)
    tf.summary.scalar(var_name, mean)
    tf.summary.histogram(var_name, var)


def summary_loss(var):
    scope_str = str(var.name)[:-2]
    with tf.name_scope(scope_str):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


def summary_image(var, max_outputs=0, var_name='image'):
    tf.summary.image(var_name, var, max_outputs=max_outputs)


def summary_filter(vars_, max_outputs=0):
    pass
