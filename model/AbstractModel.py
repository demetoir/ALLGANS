from util.Logger import Logger
from dict_keys.model_metadata_keys import *
import tensorflow as tf
import traceback
import sys


class FailLoadModelError(BaseException):
    pass


class AbstractModel:
    """Abstract class of model for tensorflow graph

    TODO add docstring

    """
    VERSION = 1.0
    AUTHOR = 'demetoir'

    def __str__(self):
        return "%s_%s_%.1f" % (self.AUTHOR, self.__class__.__name__, self.VERSION)

    def __init__(self, logger_path=None):
        """create instance of AbstractModel

        :type logger_path: str
        :param logger_path: path for log file
        if logger_path is None, log ony stdout
        """
        if logger_path is None:
            self.logger = Logger(self.__class__.__name__, with_file=True)
        else:
            self.logger = Logger(self.__class__.__name__, logger_path)
        self.log = self.logger.get_log()

    def load_model(self, metadata=None, input_shapes=None, params=None):
        """load tensor graph of entire model

        load model instance and inject metadata and input_shapes

        :param params:
        :type metadata: dict
        :type input_shapes: dict
        :param metadata: metadata for model
        :param input_shapes: input shapes for tensorflow placeholder
        :param params:

        :raise FailLoadModelError
        if any Error raise while load model
        """
        try:
            self.log("load metadata")
            self.load_metadata(metadata)

            with tf.variable_scope("misc_ops"):
                self.log('load misc ops')
                self.load_misc_ops()

            with tf.variable_scope("hyper_parameter"):
                if params is None:
                    params = self.params
                self.log('load hyper parameter')
                self.load_hyper_parameter(params)

            if input_shapes is None:
                input_shapes = self.input_shapes
            self.log("load input shapes")
            self.load_input_shapes(input_shapes)

            self.log('load main tensor graph')
            self.load_main_tensor_graph()

            with tf.variable_scope('loss'):
                self.log('load loss')
                self.load_loss_function()

            with tf.variable_scope('train_ops'):
                self.log('load train ops')
                self.load_train_ops()

            with tf.variable_scope('summary_ops'):
                self.log('load summary load')
                self.load_summary_ops()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.log("\n", "".join(traceback.format_tb(exc_traceback)))
            raise FailLoadModelError("fail to load model")
        else:
            self.log("load model complete")

    def load_metadata(self, metadata=None):
        """load metadata

        :type metadata: dict
        :param metadata: metadata for model
        """
        if metadata is None:
            self.log('skip to load metadata')
            return

        self.instance_id = metadata[MODEL_METADATA_KEY_INSTANCE_ID]
        self.instance_path = metadata[MODEL_METADATA_KEY_INSTANCE_PATH]
        self.instance_visual_result_folder_path = metadata[MODEL_METADATA_KEY_INSTANCE_VISUAL_RESULT_FOLDER_PATH]
        self.instance_source_path = metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH]
        self.instance_class_name = metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME]
        self.readme = metadata[MODEL_METADATA_KEY_README]
        self.instance_summary_folder_path = metadata[MODEL_METADATA_KEY_INSTANCE_SUMMARY_FOLDER_PATH]
        self.params = metadata[MODEL_METADATA_KEY_PARAMS]
        self.input_shapes = metadata[MODEL_METADATA_KEY_INPUT_SHAPES]

    def load_input_shapes(self, input_shapes):
        """load input shapes for tensor placeholder

        :type input_shapes: dict
        :param input_shapes: input shapes for tensor placeholder

        :raise NotImplementError
        if not Implemented
        """
        raise NotImplementedError

    def load_hyper_parameter(self, params=None):
        """load hyper parameter for model

        :param params:
        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def load_main_tensor_graph(self):
        """load main tensor graph

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def load_loss_function(self):
        """load loss function of model

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def load_misc_ops(self):
        """load misc operation of model

        :raise NotImplementError
        if not implemented
        """
        with tf.variable_scope('misc_ops'):
            self.global_step = tf.get_variable("global_step", shape=1, initializer=tf.zeros_initializer)
            with tf.variable_scope('op_inc_global_step'):
                self.op_inc_global_step = self.global_step.assign(self.global_step + 1)

    def load_train_ops(self):
        """Load train operation of model

        :raise NotImplementError
        if not implemented
        """
        raise NotImplementedError

    def load_summary_ops(self):
        """load summary operation for tensorboard

        :raise NotImplemented
        if not implemented
        """
        raise NotImplementedError

    def train_model(self, sess=None, iter_num=None, dataset=None):
        """train model

        :type sess: Session object for tensorflow.Session
        :type iter_num: int
        :type dataset: AbstractDataset
        :param sess: session object for tensorflow
        :param iter_num: current iteration number
        :param dataset: dataset for train model

        :raise NotImplemented
        if not implemented
        """
        raise NotImplementedError

    def write_summary(self, sess=None, iter_num=None, dataset=None, summary_writer=None):
        """write summary of model for tensorboard

        :type sess: Session object for tensorflow.Session
        :type iter_num: int
        :type dataset: dataset_handler.AbstractDataset
        :type summary_writer: tensorflow.summary.FileWriter

        :param sess: session object for tensorflow
        :param iter_num: current iteration number
        :param dataset: dataset for train model
        :param summary_writer: file writer for tensorboard summary

        :raise NotImplementedError
        if not implemented
        """
        raise NotImplementedError
