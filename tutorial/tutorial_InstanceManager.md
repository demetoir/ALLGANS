# Tutorial - InstanceManager

## Detail
1. build instance from model class
2. load instance
3. load visualizer
4. train instance or sample instance

## Usage step

### build instance

    ```python
    from ModelClassLoader import ModelClassLoader
    from InstanceManger import InstanceManager

    model = ModelClassLoader.load_model_class("model")
    manager = InstanceManager()
    model_metadata_path = manager.build_instance(model)
    ```

### load Instance

    ```python
    # after build model
    manager.load_instance(model_metadata_path, input_shapes_from_dataset)
    ```

### load Visualizer

```python
from VisualizerClassLoader import VisualizerClassLoader

visualizer = VisualizerClassLoader.load_class("visualizer_name")
manager.load_visualizer(visualizer, execute_interval=10)
```

### train Instance

    ```python
    # after load instance into manager train model
    manager.train_instance(
            epoch_time,
        dataset=dataset,
        check_point_interval=check_point_interval,
        # if need to start train from checkpoint
        # set is_restore to True
        # is_restore=True

        # default to start with tensorboard sub process while train instance and after train instance close tensorboard
        # if does not need tensorboard set with_tensorboard to False
        # with_tensorboard=true
    )
    ```

### sample Instance
    ```python
    # after load instance
    manager.sampling_instance(
        dataset=dataset
    )
    ```
