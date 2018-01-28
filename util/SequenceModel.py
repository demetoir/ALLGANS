import tensorflow as tf


class SequenceModel:
    def __init__(self, start_layer=None, reuse=False, name="seqmodel"):
        self.reuse = reuse
        self.layer_count = 1
        self.last_layer = start_layer
        self.layer_seq = [start_layer]
        self.name = name

    def add_layer(self, func, *args, **kwargs):
        scope_name = self.name + '_layer' + str(self.layer_count)
        with tf.variable_scope(scope_name, reuse=self.reuse):
            self.last_layer = func(self.last_layer, *args, **kwargs)
            self.layer_seq += [self.last_layer]
            pass
        self.layer_count += 1
        return self.last_layer
