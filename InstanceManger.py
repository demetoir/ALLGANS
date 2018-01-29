from shutil import copy
from time import strftime, localtime
from util.Logger import Logger
from util.util import load_class_from_source_path
from dict_keys.model_metadata_keys import *
from env_settting import tensorboard_dir

import sys
import tensorflow as tf
import os
import inspect
import json
import subprocess

TOTAL_EPOCH = 10000
CHECK_POINT_INTERVAL = 1000

META_DATA_FILE_NAME = 'model.meta'
INSTANCE_FOLDER = 'instance'
VISUAL_RESULT_FOLDER = 'visual_results'


def log_exception(func):
    def wrapper(*args):
        try:
            return func(*args)
        except Exception as e:
            self = args[0]
            self.log(str(e))
            self.log('catch error')

    return wrapper


class InstanceManager:
    def __init__(self, root_path):
        self.root_path = root_path
        self.logger = Logger(self.__class__.__name__, self.root_path)
        self.log = self.logger.get_log()
        self.model = None
        self.visualizers = []
        self.sub_process = {}

    def __del__(self):
        # reset tensorflow graph
        tf.reset_default_graph()

        for process_name in self.sub_process:
            if self.sub_process[process_name].poll is None:
                self.close_subprocess(process_name)

        del self.root_path
        del self.log
        del self.logger
        del self.model
        del self.visualizers

    def gen_instance(self, model=None, input_shapes=None):
        # gen instance id
        model_name = "%s_%s_%.1f" % (model.AUTHOR, model.__name__, model.VERSION)
        instance_id = model_name + '_' + strftime("%Y-%m-%d_%H-%M-%S", localtime())
        self.log('gen instance id : %s' % instance_id)

        # init instance directory
        instance_path = os.path.join(self.root_path, INSTANCE_FOLDER)
        if not os.path.exists(instance_path):
            os.mkdir(instance_path)

        # init user instance directory    
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
        self.log('init instance directory')

        # copy model's module file to instance/src/"model_id.py"
        instance_source_path = os.path.join(instance_source_folder_path, instance_id + '.py')
        try:
            copy(inspect.getsourcefile(model), instance_source_path)
        except IOError as e:
            print(e)
        self.log('dump model source code')

        # init and dump metadata
        metadata = {
            MODEL_METADATA_KEY_INSTANCE_ID: instance_id,
            MODEL_METADATA_KEY_INSTANCE_PATH: instance_path,
            MODEL_METADATA_KEY_INSTANCE_VISUAL_RESULT_FOLDER_PATH: instance_visual_result_folder_path,
            MODEL_METADATA_KEY_INSTANCE_SOURCE_FOLDER: instance_source_folder_path,
            MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH: instance_source_path,
            MODEL_METADATA_KEY_INSTANCE_CLASS_NAME: model.__name__,
            MODEL_METADATA_KEY_README: self.gen_readme(),
            MODEL_METADATA_KEY_INSTANCE_SUMMARY_FOLDER_PATH: instance_summary_folder_path
        }
        metadata_path = os.path.join(instance_path, 'instance.meta')
        self.dump_json(metadata, metadata_path)
        self.log('dump metadata')

        # build model
        self.model = model(metadata, input_shapes)
        self.log('build model')

        self.metadata_path = metadata_path
        self.metadata = metadata

    def load_model(self, metadata_path, input_shapes):
        metadata = self.load_json(metadata_path)
        self.log('load metadata')

        instance_class_name = metadata[MODEL_METADATA_KEY_INSTANCE_CLASS_NAME]
        instance_source_path = metadata[MODEL_METADATA_KEY_INSTANCE_SOURCE_PATH]
        model = load_class_from_source_path(instance_source_path, instance_class_name)
        self.log('model source code load')

        self.model = model(metadata, input_shapes)
        self.log('build model')

        instance_id = metadata[MODEL_METADATA_KEY_INSTANCE_ID]
        self.log('load instance id : %s' % instance_id)

    def load_visualizer(self, visualizers):
        visualizer_path = self.model.instance_visual_result_folder_path
        for visualizer, iter_cycle in visualizers:
            if not os.path.exists(visualizer_path):
                os.mkdir(visualizer_path)

            self.visualizers += [visualizer(visualizer_path, iter_cycle=iter_cycle)]
            self.log('visualizer %s loaded' % visualizer.__name__)

        self.log('visualizer fully Load')

    def train_model(self, dataset, epoch_time=TOTAL_EPOCH, check_point_interval=CHECK_POINT_INTERVAL, is_restore=False):
        saver = tf.train.Saver()
        save_path = os.path.join(self.model.instance_path, 'check_point')
        check_point_path = os.path.join(save_path, 'model.ckpt')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            self.log('make save dir')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.log('init global variables')

            summary_writer = tf.summary.FileWriter(self.model.instance_summary_folder_path, sess.graph)
            self.log('init summary_writer')

            if is_restore:
                saver.restore(sess, check_point_path)
                self.log('restore check point')

            batch_size = self.model.batch_size
            iter_per_epoch = int(dataset.data_size / batch_size)
            self.log('total Epoch: %d, total iter: %d, iter per epoch: %d' % (
                epoch_time, epoch_time * iter_per_epoch, iter_per_epoch))

            iter_num, loss_val_D, loss_val_G = 0, 0, 0
            for epoch in range(epoch_time):
                for _ in range(iter_per_epoch):
                    iter_num += 1
                    self.model.train_model(sess=sess, iter_num=iter_num, dataset=dataset)
                    self.__visualizer_task(sess, iter_num, dataset)

                    self.model.write_summary(sess=sess, iter_num=iter_num, dataset=dataset,
                                             summary_writer=summary_writer)

                    if iter_num % check_point_interval == 0:
                        saver.save(sess, check_point_path)

        self.log('train end')

        tf.reset_default_graph()
        self.log('reset default graph')

    def sample_model(self, is_restore=False):
        self.log('start train_model')
        saver = tf.train.Saver()

        save_path = os.path.join(self.model.instance_path, 'check_point')
        check_point_path = os.path.join(save_path, 'model.ckpt')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            self.log('make save dir')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_restore:
                saver.restore(sess, check_point_path)
                self.log('restore check point')

            self.__visualizer_task(sess)

        self.log('sampling end')

        tf.reset_default_graph()
        self.log('reset default graph')

    def __visualizer_task(self, sess, iter_num=None, dataset=None):
        for visualizer in self.visualizers:
            if iter_num is None or iter_num % visualizer.iter_cycle == 0:
                try:
                    visualizer.task(sess, iter_num, self.model, dataset)
                except Exception as err:
                    self.log('at visualizer %s \n %s' % (visualizer, err))

    def open_subprocess(self, args, process_name):
        if process_name in self.sub_process and self.sub_process[process_name].poll is not None:
            # TODO better error class
            raise AssertionError("process '%s'(pid:%s) already exist and still running" % (
                process_name, self.sub_process[process_name].pid))

        self.sub_process[process_name] = subprocess.Popen(args)
        str_args = " ".join(map(str, args))
        pid = self.sub_process[process_name].pid
        self.log("open subprocess '%s',  pid: %s" % (str_args, pid))

    def close_subprocess(self, process_name):
        if process_name in self.sub_process:
            self.log("kill subprocess '%s', pid: %s" % (process_name, self.sub_process[process_name].pid))
            self.sub_process[process_name].kill()
        else:
            raise KeyError("fail close subprocess, '%s' not found" % process_name)

    def open_tensorboard(self):
        python_path = sys.executable
        option = '--logdir=' + self.model.instance_summary_folder_path
        args = [python_path, tensorboard_dir(), option]
        self.open_subprocess(args=args, process_name="tensorboard")

    def close_tensorboard(self):
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
