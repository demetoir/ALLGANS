from InstanceManger import InstanceManager


class InstanceManagerHelper:
    @staticmethod
    def build_and_train(model=None, input_shapes=None, dataset=None, visualizers=None, env_path=None, epoch_time=50,
                        check_point_interval_per_iter=5000):
        manager = InstanceManager(env_path)
        metadata_path = manager.build_instance(model)
        manager.load_instance(metadata_path, input_shapes)
        manager.load_visualizer(visualizers)

        manager.open_tensorboard()
        manager.train_instance(dataset, epoch_time, check_point_interval_per_iter)
        manager.close_tensorboard()
        del manager

    @staticmethod
    def build_models(model_list=None, input_shapes=None, env_setting=None):
        for model in model_list:
            manager = InstanceManager(env_setting)
            manager.build_instance(model, input_shapes)
            del manager

    @staticmethod
    def load_and_train(input_shapes=None, visualizers=None, metadata_path=None, env_setting=None, dataset=None,
                       epoch_time=50, check_point_interval_per_iter=5000):
        manager = None
        try:
            manager = InstanceManager(env_setting)
            manager.load_instance(instance_path=metadata_path, input_shapes=input_shapes)
            manager.load_visualizer(visualizers)

            manager.open_tensorboard()

            manager.train_instance(dataset, epoch_time, check_point_interval_per_iter, is_restore=True)
        except Exception as e:
            print(e)
        finally:
            manager.close_tensorboard()
            del manager

    @staticmethod
    def load_and_sampling(input_shapes=None, visualizers=None, metadata_path=None, env_setting=None):
        manager = InstanceManager(env_setting)
        manager.load_instance(instance_path=metadata_path, input_shapes=input_shapes)
        manager.load_visualizer(visualizers)
        manager.sampling_instance(is_restore=True)
