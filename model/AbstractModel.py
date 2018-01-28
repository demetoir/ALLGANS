from util.Logger import Logger
from dict_keys.model_metadata_keys import *
import tensorflow as tf


class AbstractModel:
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __str__(self):
        return "%s_%s_%.1f" % (self.AUTHOR, self.__class__.__name__, self.VERSION)

    def __init__(self, metadata, input_shapes):
        self.logger = Logger(self.__class__.__name__, metadata[MODEL_METADATA_KEY_INSTANCE_PATH])
        self.log = self.logger.get_log()

        self.load_meta_data(metadata)
        self.input_shapes(input_shapes)
        self.hyper_parameter()
        self.network()
        self.log('network build')
        self.loss()
        self.log('loss build')
        self.train_ops()
        self.log('train ops build')
        self.misc_ops()
        self.log('misc ops build')
        self.summary_op()
        self.log('summary build')

    def load_meta_data(self, metadata):
        self.instance_id = metadata[MODEL_METADATA_KEY_INSTANCE_ID]
        self.instance_path = metadata[MODEL_METADATA_KEY_INSTANCE_PATH]
        self.instance_visual_result_folder_path = metadata[MODEL_METADATA_KEY_INSTANCE_VISUAL_RESULT_FOLDER_PATH]
        self.instance_source_path = metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH]
        self.instance_class_name = metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME]
        self.readme = metadata[MODEL_METADATA_KEY_README]
        self.instance_summary_folder_path = metadata[MODEL_METADATA_KEY_INSTANCE_SUMMARY_FOLDER_PATH]

    def input_shapes(self, input_shapes):
        raise NotImplementedError

    def hyper_parameter(self):
        raise NotImplementedError

    def network(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def train_ops(self):
        raise NotImplementedError

    def misc_ops(self):
        # TODO scope problem ..?
        with tf.variable_scope('misc_ops'):
            self.global_step = tf.get_variable("global_step", shape=[1], initializer=tf.zeros_initializer)
            with tf.variable_scope('op_inc_global_step'):
                self.op_inc_global_step = self.global_step.assign(self.global_step + 1)

    def train_model(self, sess=None, iter_num=None, dataset=None):
        raise NotImplementedError

    def summary_op(self):
        raise NotImplementedError

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        raise NotImplementedError
