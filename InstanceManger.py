from dict_keys.model_metadata_keys import *
from model.AbstractModel import AbstractModel
from util.Logger import Logger
from util.misc_util import import_class_from_module_path, dump_json, load_json
from env_settting import *
from shutil import copy
from time import strftime, localtime
import tensorflow as tf
import inspect
import os
import subprocess
import sys
import traceback

META_DATA_FILE_NAME = 'instance.meta'
INSTANCE_FOLDER = 'instance'
VISUAL_RESULT_FOLDER = 'visual_results'


def deco_handle_exception(func):
    """decorator for catch exception and log
    """

    def wrapper(*args, **kwargs):
        self = args[0]
        log_func = self.log
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            log_func("KeyboardInterrupt detected abort process")
        except Exception as e:
            log_error_trace(log_func, e)

    return wrapper


def log_error_trace(log_func, e, head=""):
    exc_type, exc_value, exc_traceback = sys.exc_info()

    msg = '%s\n %s %s : %s \n' % (
        head,
        "".join(traceback.format_tb(exc_traceback)),
        e.__class__.__name__,
        e,
    )
    log_func(msg)


class InstanceManager:
    """ manager class for Instance

    step for managing instance
    1. build instance(if already built instance ignore this step)
    2. load_instance and visualizers
    3. train_instance or sampling_instance

    ex)for build instance and train
    manager = InstanceManager(env_path)
    instance_path = manager.build_instance(model)
    manager.load_instance(instance_path, input_shapes)
    manager.load_visualizer(visualizers)
    manager.train_instance(dataset, epoch, check_point_interval)

    ex) resume from training
    manager = InstanceManager(env_path)
    manager.load_instance(built_instance_path, input_shapes)
    manager.load_visualizer(visualizers)
    manager.train_instance(dataset, epoch, check_point_interval, is_restore=True)
    """

    def __init__(self, root_path=ROOT_PATH):
        """ create a 'InstanceManager' at env_path

        :type root_path: str
        :param root_path: env path for manager
        """
        self.root_path = root_path
        self.logger = Logger(self.__class__.__name__, self.root_path)
        self.log = self.logger.get_log()
        self.instance = None
        self.visualizers = {}
        self.subprocess = {}

    def __del__(self):
        """ destructor of InstanceManager

        clean up all memory, subprocess, logging, tensorflow graph
        """
        # reset tensorflow graph
        tf.reset_default_graph()

        for process_name in self.subprocess:
            if self.subprocess[process_name].poll is None:
                self.close_subprocess(process_name)

        del self.root_path
        del self.log
        del self.logger
        del self.instance
        del self.visualizers

    def build_instance(self, model=None, input_shapes=None, param=None):
        """build instance for model class and return instance path

        * model must be subclass of AbstractModel

        generate unique id to new instance and initiate folder structure
        dump model's script
        generate and save metadata for new instance
        return built instance's path

        :param input_shapes:
        :param param:
        :type model: class
        :param model: subclass of AbstractModel

        :return: built instance's path
        """
        if not issubclass(model, AbstractModel):
            raise TypeError("argument model expect subclass of AbstractModel")

        # gen instance id
        model_name = "%s_%s_%.1f" % (model.AUTHOR, model.__name__, model.VERSION)
        instance_id = model_name + '_' + strftime("%Y-%m-%d_%H-%M-%S", localtime())
        self.log('build instance: %s' % instance_id)

        # init new instance directory
        self.log('init instance directory')
        instance_path = os.path.join(self.root_path, INSTANCE_FOLDER, instance_id)
        if not os.path.exists(instance_path):
            os.mkdir(instance_path)

        instance_visual_result_folder_path = os.path.join(instance_path, VISUAL_RESULT_FOLDER)
        if not instance_visual_result_folder_path:
            os.mkdir(instance_visual_result_folder_path)

        instance_source_folder_path = os.path.join(instance_path, 'src_code')
        if not os.path.exists(instance_source_folder_path):
            os.mkdir(instance_source_folder_path)

        instance_summary_folder_path = os.path.join(instance_path, 'summary')
        if not os.path.exists(instance_summary_folder_path):
            os.mkdir(instance_summary_folder_path)

        self.log('dump instance source code')
        instance_source_path = os.path.join(instance_source_folder_path, instance_id + '.py')
        try:
            copy(inspect.getsourcefile(model), instance_source_path)
        except IOError as e:
            print(e)

        self.log("build_metadata")
        metadata_path = os.path.join(instance_path, 'instance.meta')
        metadata = {
            MODEL_METADATA_KEY_INSTANCE_ID: instance_id,
            MODEL_METADATA_KEY_INSTANCE_PATH: instance_path,
            MODEL_METADATA_KEY_INSTANCE_VISUAL_RESULT_FOLDER_PATH: instance_visual_result_folder_path,
            MODEL_METADATA_KEY_INSTANCE_SOURCE_FOLDER_PATH: instance_source_folder_path,
            MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH: instance_source_path,
            MODEL_METADATA_KEY_INSTANCE_SUMMARY_FOLDER_PATH: instance_summary_folder_path,
            MODEL_METADATA_KEY_INSTANCE_CLASS_NAME: model.__name__,
            MODEL_METADATA_KEY_README: None,
            MODEL_METADATA_KEY_METADATA_PATH: metadata_path,
            MODEL_METADATA_KEY_PARAMS: param,
            MODEL_METADATA_KEY_INPUT_SHAPES: input_shapes,
        }

        self.log('dump metadata')
        dump_json(metadata, metadata_path)

        self.log('build complete')
        return instance_path

    def load_instance(self, instance_path):
        """ load built instance into InstanceManager

        import model class from dumped script in instance_path
        inject metadata and input_shapes into model
        load tensorflow graph from model
        load instance into InstanceManager

        * more information for input_shapes look dict_keys/input_shape_keys.py

        :type instance_path: str
        :param instance_path: instance path to loading
        """
        metadata = load_json(os.path.join(instance_path, 'instance.meta'))
        self.log('load metadata')

        instance_class_name = metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME]
        instance_source_path = metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH]
        model = import_class_from_module_path(instance_source_path, instance_class_name)
        self.log('instance source code load')

        self.instance = model(metadata[MODEL_METADATA_KEY_INSTANCE_PATH])
        self.instance.load_model(metadata)
        self.log('load instance')

        instance_id = metadata[MODEL_METADATA_KEY_INSTANCE_ID]
        self.log('load instance id : %s' % instance_id)

    @deco_handle_exception
    def train_instance(self, epoch, dataset=None, check_point_interval=None, is_restore=False, with_tensorboard=True):
        """training loaded instance with dataset for epoch and loaded visualizers will execute

        * if you want to use visualizer call load_visualizer function first

        every check point interval, tensor variables will save at check point
        check_point_interval's default is one epoch, but scale of interval is number of iteration
        so if check_point_interval=3000, tensor variable save every 3000 per iter
        option is_restore=False is default
        if you want to restore tensor variables from check point, use option is_restore=True

        InstanceManager may open subprocess like tensorboard, raising error may cause some issue
        like subprocess still alive, while InstanceManager process exit
        so any error raise while training wrapper @log_exception will catch error
        KeyboardInterrupt raise, normal exit for abort training and return
        any other error will print error message and return

        :param epoch: total epoch for train
        :param dataset: dataset for train
        :param check_point_interval: interval for check point to save train tensor variables
        :param is_restore: option for restoring from check point
        :param with_tensorboard: option for open child process for tensorboard to monitor summary
        """

        if with_tensorboard:
            self.open_tensorboard()

        self.log("current loaded visualizers")
        for key in self.visualizers:
            self.log(key)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            save_path = os.path.join(self.instance.instance_path, 'check_point')
            check_point_path = os.path.join(save_path, 'instance.ckpt')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                self.log('make save dir')

            self.log('init global variables')
            sess.run(tf.global_variables_initializer())

            self.log('init summary_writer')
            summary_writer = tf.summary.FileWriter(self.instance.instance_summary_folder_path, sess.graph)

            if is_restore:
                self.log('restore check point')
                saver.restore(sess, check_point_path)

            batch_size = self.instance.batch_size
            iter_per_epoch = int(dataset.train_set.data_size / batch_size)
            self.log('train set size: %d, total Epoch: %d, total iter: %d, iter per epoch: %d'
                     % (dataset.train_set.data_size, epoch, epoch * iter_per_epoch, iter_per_epoch))

            iter_num = 0
            for epoch_ in range(epoch):
                # TODO need concurrency
                dataset.shuffle()
                for _ in range(iter_per_epoch):
                    iter_num += 1
                    self.instance.train_model(sess=sess, iter_num=iter_num, dataset=dataset)
                    self.__visualizer_task(sess, iter_num, dataset)

                    self.instance.write_summary(sess=sess, iter_num=iter_num, dataset=dataset,
                                                summary_writer=summary_writer)

                    if iter_num % check_point_interval == 0:
                        saver.save(sess, check_point_path)
                # self.log("epoch %s end" % (epoch_ + 1))

            saver.save(sess, check_point_path)
        self.log('train end')

        if with_tensorboard:
            self.close_tensorboard()

    @deco_handle_exception
    def sampling_instance(self, dataset=None, is_restore=True):
        """sampling result from trained instance by running loaded visualizers

        * if you want to use visualizer call load_visualizer function first

        InstanceManager may open subprocess like tensorboard, raising error may cause some issue
        like subprocess still alive, while InstanceManager process exit
        so any error raise while training wrapper @log_exception will catch error
        KeyboardInterrupt raise, normal exit for abort training and return
        any other error will print error message and return

        :param dataset:
        :param is_restore: option for restoring from check point
        """
        self.log('start sampling_model')
        saver = tf.train.Saver()

        self.log("current loaded visualizers")
        for key in self.visualizers:
            self.log(key)

        save_path = os.path.join(self.instance.instance_path, 'check_point')
        check_point_path = os.path.join(save_path, 'instance.ckpt')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_restore:
                saver.restore(sess, check_point_path)
                self.log('restore check point')

            iter_num = 0
            for visualizer in self.visualizers.values():
                try:
                    visualizer.task(sess=sess, iter_num=iter_num, model=self.instance, dataset=dataset)
                except Exception as err:
                    log_error_trace(self.log, err, head='while execute %s' % visualizer)

        self.log('sampling end')

    def load_visualizer(self, visualizer, execute_interval, key=None):
        """load visualizer for training and sampling result of instance

        :type visualizer: AbstractVisualizer
        :param visualizer: list of tuple,
        :type execute_interval: int
        :param execute_interval: interval to execute visualizer per iteration
        :param key: key of visualizer dict
        """
        visualizer_path = self.instance.instance_visual_result_folder_path
        if not os.path.exists(visualizer_path):
            os.mkdir(visualizer_path)

        if key is None:
            key = visualizer.__name__

        self.visualizers[key] = visualizer(visualizer_path, execute_interval=execute_interval)
        self.log('visualizer %s loaded key=%s' % (visualizer.__name__, key))
        return key

    def unload_visualizer(self, key):
        if key not in self.visualizers:
            raise KeyError("fail to unload visualizer, key '%s' not found" % key)
        self.visualizers.pop(key, None)

    def unload_all_visualizer(self):
        for key in self.visualizers:
            self.visualizers.pop(key, None)

    def __visualizer_task(self, sess, iter_num=None, dataset=None):
        """execute loaded visualizers

        :type iter_num: int
        :type dataset: AbstractDataset
        :param sess: tensorflow.Session object
        :param iter_num: current iteration number
        :param dataset: feed for visualizers
        """
        for visualizer in self.visualizers.values():
            if iter_num is None or iter_num % visualizer.execute_interval == 0:
                try:
                    visualizer.task(sess, iter_num, self.instance, dataset)
                except Exception as err:
                    log_error_trace(self.log, err, head='while execute %s' % visualizer)

    def open_subprocess(self, args_, subprocess_key=None):
        """open subprocess with args and return pid

        :type args_: list
        :type subprocess_key: str
        :param args_: list of argument for new subprocess
        :param subprocess_key: key for self.subprocess of opened subprocess
        if subprocess_key is None, pid will be subprocess_key

        :raise ChildProcessError
        if same process name is already opened

        :return: pid for opened subprocess
        """

        if subprocess_key in self.subprocess and self.subprocess[subprocess_key].poll is not None:
            # TODO better error class

            raise AssertionError("process '%s'(pid:%s) already exist and still running" % (
                subprocess_key, self.subprocess[subprocess_key].pid))

        child_process = subprocess.Popen(args_)
        if subprocess_key is None:
            subprocess_key = str(child_process.pid)
        self.subprocess[subprocess_key] = child_process
        str_args = " ".join(map(str, args_))
        self.log("open subprocess pid:%s, cmd='%s'" % (child_process.pid, str_args))

        return child_process.pid

    def close_subprocess(self, subprocess_key):
        """close subprocess

        close opened subprocess of process_name

        :type subprocess_key: str
        :param subprocess_key: key for closing subprocess

        :raises KeyError
        if subprocess_key is not key for self.subprocess
        """
        if subprocess_key in self.subprocess:
            self.log("kill subprocess pid:%s, '%s'" % (self.subprocess[subprocess_key].pid, subprocess_key))
            self.subprocess[subprocess_key].kill()
        else:
            raise KeyError("fail close subprocess, '%s' not found" % subprocess_key)

    def open_tensorboard(self):
        """open tensorboard for current instance"""
        python_path = sys.executable
        option = '--logdir=' + self.instance.instance_summary_folder_path
        # option += ' --port 6006'
        # option += ' --debugger_port 6064'
        args_ = [python_path, tensorboard_dir(), option]
        self.open_subprocess(args_=args_, subprocess_key="tensorboard")

    def close_tensorboard(self):
        """close tensorboard for current instance"""
        self.close_subprocess('tensorboard')

    def get_tf_values(self, fetches, feed_dict):
        return self.instance.get_tf_values(self.sess, fetches, feed_dict)
