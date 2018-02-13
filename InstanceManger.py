from dict_keys.model_metadata_keys import *
from util.Logger import Logger
from util.misc_util import import_class_from_module_path
from shutil import copy
from time import strftime, localtime
from env_settting import *
import tensorflow as tf
import inspect
import json
import os
import subprocess
import sys
import traceback

META_DATA_FILE_NAME = 'instance.meta'
INSTANCE_FOLDER = 'instance'
VISUAL_RESULT_FOLDER = 'visual_results'


def log_exception(func):
    """wrapper for catch exception and log
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            self = args[0]
            self.log("KeyboardInterrupt detected abort process")
            print("KeyboardInterrupt detected abort process")
        except Exception as e:
            self = args[0]
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.log("\n", "".join(traceback.format_tb(exc_traceback)))

    return wrapper


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
        self.visualizers = []
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

    def build_instance(self, model=None):
        """build instance for model class and return instance path

        * model must be subclass of AbstractModel

        generate unique id to new instance and initiate folder structure
        dump model's script
        generate and save metadata for new instance
        return built instance's path

        :type model: class
        :param model: subclass of AbstractModel

        :return: built instance's path
        """
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

        # init and dump metadata
        self.log("build_metadata")
        metadata = model.build_metadata()

        self.log('dump metadata')
        metadata[MODEL_METADATA_KEY_INSTANCE_ID] = instance_id
        metadata[MODEL_METADATA_KEY_INSTANCE_PATH] = instance_path
        metadata[MODEL_METADATA_KEY_INSTANCE_VISUAL_RESULT_FOLDER_PATH] = instance_visual_result_folder_path
        metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_FOLDER_PATH] = instance_source_folder_path
        metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH] = instance_source_path
        metadata[MODEL_METADATA_KEY_INSTANCE_SUMMARY_FOLDER_PATH] = instance_summary_folder_path
        metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME] = model.__name__
        metadata[MODEL_METADATA_KEY_README] = self.gen_readme()
        metadata[MODEL_METADATA_KEY_METADATA_PATH] = os.path.join(instance_path, 'instance.meta')
        self.dump_json(metadata, metadata[MODEL_METADATA_KEY_METADATA_PATH])

        self.log('build complete')
        return instance_path

    def load_instance(self, instance_path, input_shapes):
        """ load built instance into InstanceManager

        import model class from dumped script in instance_path
        inject metadata and input_shapes into model
        load tensorflow graph from model
        load instance into InstanceManager

        * more information for input_shapes look dict_keys/input_shape_keys.py

        :type instance_path: str
        :type input_shapes: dict
        :param instance_path: instance path to loading
        :param input_shapes: input shapes for tensorflow placeholder
        """
        metadata = self.load_json(os.path.join(instance_path, 'instance.meta'))
        self.log('load metadata')

        instance_class_name = metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME]
        instance_source_path = metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH]
        model = import_class_from_module_path(instance_source_path, instance_class_name)
        self.log('instance source code load')

        self.instance = model(metadata[MODEL_METADATA_KEY_INSTANCE_PATH])
        self.instance.load_model(metadata, input_shapes)
        self.log('load instance')

        instance_id = metadata[MODEL_METADATA_KEY_INSTANCE_ID]
        self.log('load instance id : %s' % instance_id)

    @log_exception
    def train_instance(self, epoch, dataset=None, check_point_interval=None, is_restore=False):
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
        """
        saver = tf.train.Saver()
        save_path = os.path.join(self.instance.instance_path, 'check_point')
        check_point_path = os.path.join(save_path, 'instance.ckpt')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            self.log('make save dir')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.log('init global variables')

            summary_writer = tf.summary.FileWriter(self.instance.instance_summary_folder_path, sess.graph)
            self.log('init summary_writer')

            if is_restore:
                saver.restore(sess, check_point_path)
                self.log('restore check point')

            batch_size = self.instance.batch_size
            print(dataset)
            iter_per_epoch = int(dataset.data_size / batch_size)
            self.log('total Epoch: %d, total iter: %d, iter per epoch: %d' % (
                epoch, epoch * iter_per_epoch, iter_per_epoch))

            iter_num, loss_val_D, loss_val_G = 0, 0, 0
            for epoch_ in range(epoch):
                for _ in range(iter_per_epoch):
                    iter_num += 1
                    self.instance.train_model(sess=sess, iter_num=iter_num, dataset=dataset)
                    self.__visualizer_task(sess, iter_num, dataset)

                    self.instance.write_summary(sess=sess, iter_num=iter_num, dataset=dataset,
                                                summary_writer=summary_writer)

                    if iter_num % check_point_interval == 0:
                        saver.save(sess, check_point_path)
                self.log("epoch %s end" % (epoch_ + 1))
        self.log('train end')

        tf.reset_default_graph()
        self.log('reset default graph')

    @log_exception
    def sampling_instance(self, dataset=None, is_restore=False):
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

        save_path = os.path.join(self.instance.instance_path, 'check_point')
        check_point_path = os.path.join(save_path, 'instance.ckpt')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_restore:
                saver.restore(sess, check_point_path)
                self.log('restore check point')

            self.__visualizer_task(sess, dataset=dataset)
        self.log('sampling end')

        tf.reset_default_graph()
        self.log('reset default graph')

    def load_visualizer(self, visualizers):
        """load visualizers for training and sampling result of instance

        visualizers is list of tuples, each tuple is contain visualizer class and execute iter interval
        ex)
        visualizers = [(visualizer1, execute_interval1),
            [(visualizer2, execute_interval2),
            [(visualizer3, execute_interval3),]

        :type visualizers: list
        :param visualizers: list of tuple,
        :return:
        """
        visualizer_path = self.instance.instance_visual_result_folder_path
        for visualizer, execute_interval in visualizers:
            if not os.path.exists(visualizer_path):
                os.mkdir(visualizer_path)

            self.visualizers += [visualizer(visualizer_path, execute_interval=execute_interval)]
            self.log('visualizer %s loaded' % visualizer.__name__)

        self.log('visualizer fully Load')

    def __visualizer_task(self, sess, iter_num=None, dataset=None):
        """execute loaded visualizers

        :type iter_num: int
        :type dataset: AbstractDataset
        :param sess: tensorflow.Session object
        :param iter_num: current iteration number
        :param dataset: feed for visualizers
        """
        for visualizer in self.visualizers:
            if iter_num is None or iter_num % visualizer.execute_interval == 0:
                try:
                    visualizer.task(sess, iter_num, self.instance, dataset)
                except Exception as err:
                    self.log('at visualizer %s \n %s' % (visualizer, err))

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
        args_ = [python_path, tensorboard_dir(), option]
        self.open_subprocess(args_=args_, subprocess_key="tensorboard")

    def close_tensorboard(self):
        """close tensorboard for current instance"""
        self.close_subprocess('tensorboard')

    @staticmethod
    def dump_json(obj, path):
        with open(path, 'w') as f:
            json.dump(obj, f)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata

    @staticmethod
    def gen_readme():
        # TODO implement
        return {}
