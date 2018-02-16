# example of implementing data handler

# implement step
1. add dataset folder path in env_setting.py

```python
EXAMPLE_DATASET_PATH = os.path.join(DATA_PATH, 'example_dataset')
```

2. add dataset_batch_keys in dict_keys.dataset_batch_keys.py
```python

```

3. add input_shapes_keys in dict_keys.dataset_batch_keys.py
```python
```

4. implement dataset class in data_handler.dataset_name.py


    1. define class and make child class of AbstractDataset
    ```python

    ```

    2. implement 'self.__init__()'

    3. implement self.load()


5. implement datasetHelper class in data_handler.dataset_name.py
    1. implement self.preprocess()

    2. implement self.next_batch_task()

    3. implement self.load_dataset()

