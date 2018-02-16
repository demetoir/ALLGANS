from InstanceManger import InstanceManager


class InstanceManagerHelper:
    @staticmethod
    def build_and_train(model=None, input_shapes=None, dataset=None, visualizers=None, epoch_time=50,
                        check_point_interval=5000):
        manager = InstanceManager()
        metadata_path = manager.build_instance(model)
        manager.load_instance(metadata_path, input_shapes)
        for visualizer, interval in visualizers:
            manager.load_visualizer(visualizer, interval)

        manager.train_instance(epoch_time, dataset=dataset, check_point_interval=check_point_interval)
        del manager

    @staticmethod
    def build_test_model(model_list=None, env_setting=None):
        for model in model_list:
            manager = InstanceManager(env_setting)
            manager.build_instance(model)
            del manager

    @staticmethod
    def load_and_sampling(input_shapes=None, visualizers=None, metadata_path=None, env_setting=None):
        manager = InstanceManager(env_setting)
        manager.load_instance(instance_path=metadata_path, input_shapes=input_shapes)
        manager.load_visualizer(visualizers)
        manager.sampling_instance(is_restore=True)
