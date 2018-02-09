import tensorflow as tf


class LayerModel:
    """help easily make graph model and naming for tensorflow


    ex)

    todo

    """
    def __init__(self, start_layer=None, reuse=False, name="layerModel"):
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

        :param func: function for layer
        :param args: args for layer
        :param kwargs: kwargs for layer
        :return: added new layer
        """
        scope_name = self.name + '_layer' + str(self.layer_count)
        with tf.variable_scope(scope_name, reuse=self.reuse):
            self.last_layer = func(self.last_layer, *args, **kwargs)
            self.layer_seq += [self.last_layer]
            pass
        self.layer_count += 1
        return self.last_layer
