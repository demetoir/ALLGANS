# example - implementing data handler

## detail
dataset handler class for InstanceManager
* auto download dataset
* preprocess after load dataset
* preprocess after iterate mini batch


## Usage

    ```python
    from DatasetLoader import DatasetLoader

    # load dataset by calling DatasetLoader
    # input_shapes is for tensorflow.PlaceHolder's shape
    # need to build instanc e
    dataset, input_shapes = DatasetLoader().load_dataset("dataset_name")

    # apply to train model
    instanceManager.train_instance(
        epoch_time,
        dataset=dataset,
        check_point_interval=check_point_interval
    )

    ```

## Implement step

1. add dataset folder path in **env_setting.py**

    ```python
    EXAMPLE_DATASET_PATH = os.path.join(DATA_PATH, 'example_dataset')
    ```

2. add dataset_batch_keys in **dict_keys.dataset_batch_keys.py**
    ```python
    INPUT_SHAPE_KEY_DATA_X = "INPUT_SHAPE_KEY_DATA_X"
    INPUT_SHAPE_KEY_DATA_X = "INPUT_SHAPE_KEY_DATA_X"
    INPUT_SHAPE_KEY_LABEL = "INPUT_SHAPE_KEY_LABEL"
    INPUT_SHAPE_KEY_LABEL_SIZE = "INPUT_SHAPE_KEY_LABEL_SIZE"
    ```

3. add input_shapes_keys in **dict_keys.dataset_batch_keys.py**
    ```python
    BATCH_KEY_EXAMPLE_TRAIN_X = "BATCH_KEY_EXAMPLE_TRAIN_X"
    BATCH_KEY_EXAMPLE_TEST_X = "BATCH_KEY_EXAMPLE_TEST_X"
    BATCH_KEY_EXAMPLE_TRAIN_LABEL = "BATCH_KEY_EXAMPLE_TRAIN_LABEL"
    BATCH_KEY_EXAMPLE_TEST_LABEL = "BATCH_KEY_EXAMPLE_TEST_LABEL"
    ```

4. implement dataset class in **data_handler.dataset_name.py**

    1. define dataset class and make child class of **AbstractDataset**
        ```python
        import from data_handler.AbstractDataset import AbstractDataset

        class ExampleDataset(AbstractDataset):
            pass
        ```

    2. implement **self.__init__()**

    call **super().__init__()** and **init self.batch_keys**, **self.DOWNLOAD_URL**,
    **self.DOWNLOAD_FILE_NAME**, **self.extracted_data_files**

    * self.DOWNLOAD_URL: (str) url for download dataset
    * self.DOWNLOAD_FILE_NAME: (str) file name of zipped dataset
    * self.extracted_data_files: (list) list of extracted files name in dataset
    * self.batch_keys: (str) feature label of dataset,
        managing batch keys in dict_keys.dataset_batch_keys recommend


        ```python
        ...
        class ExampleDataset(AbstractDataset):
            def __init__(self, preprocess=None, batch_after_task=None):
                super().__init__(preprocess, batch_after_task)
                self.batch_keys = [
                    BATCH_KEY_TRAIN_X,
                    BATCH_KEY_TRAIN_LABEL,
                    BATCH_KEY_TEST_X,
                    BATCH_KEY_TEST_LABEL
                ]
                self.DOWNLOAD_URL = 'download_url'
                self.DOWNLOAD_FILE_NAME = 'download_file_name'
                self.extracted_data_files = [
                    "file1",
                    "file2",
                    "file3",
                    "file4",
                ]
        ```

    3. implement self.load()

    * load data to self.data(dict type)
    * key is batch_key and value is data of batch key


        ```python
        ...
        class ExamplDataset(AbstractDataset):
            ...
            def load(self, path, limit=None):
                # load train data
                files = glob(os.path.join(path, self._PATTERN_TRAIN_FILE))
                files.sort()
                for file in files:
                    with open(file, 'rb') as fo:
                        dict_ = pickle.load(fo, encoding='bytes')

                    x = dict_[self._PKCL_KEY_TRAIN_DATA]
                    self._append_data(BATCH_KEY_TRAIN_X, x)

                    label = dict_[self._PKCL_KEY_TRAIN_LABEL]
                    self._append_data(BATCH_KEY_TRAIN_LABEL, label)

                # load test data
                files = glob(os.path.join(path, self._PATTERN_TEST_FILE))
                files.sort()
                for file in files:
                    with open(file, 'rb') as fo:
                        dict_ = pickle.load(fo, encoding='bytes')

                    x = dict_[self._PKCL_KEY_TEST_DATA]
                    self._append_data(BATCH_KEY_TEST_X, x)

                    label = dict_[self._PKCL_KEY_TEST_LABEL]
                    self._append_data(BATCH_KEY_TEST_LABEL, label)
        ```

5. implement datasetHelper class in data_handler.dataset_name.py
    1. define datasetHelper class and make subclass of AbstractDatasetHelper
        ```python
        from data_handler.AbstractDataset import AbstractDatasetHelper

        class ExampleDatasetHelper(AbstractDatasetHelper):
            pass
        ```



    2. implement self.preprocess()

    * if you don't need preprocess for dataset than just skip this step
        ```python
        ...
        class ExampleDatasetHelper(AbstractDatasetHelper):
            @staticmethod
            def preprocess(dataset):
                # you can implement preprocess here for loaded dataset
                # original MNIST image size is 28*28 but need to resize 32*32
                data = dataset.data[BATCH_KEY_TRAIN_X]
                shape = data.shape
                data = np.reshape(data, [shape[0], 28, 28])
                npad = ((0, 0), (2, 2), (2, 2))
                data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
                data = np.reshape(data, [shape[0], 32, 32, 1])
                dataset.data[BATCH_KEY_TRAIN_X] = data

                data = dataset.data[BATCH_KEY_TEST_X]
                shape = data.shape
                data = np.reshape(data, [shape[0], 28, 28])
                npad = ((0, 0), (2, 2), (2, 2))
                data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
                data = np.reshape(data, [shape[0], 32, 32, 1])
                dataset.data[BATCH_KEY_TEST_X] = data
        ```


    3. implement self.next_batch_task()

        * if don't need next_batch_task, just skip this step
        * must return result of next_batch_task

        ```python
        ...
        class ExampleDatasetHelper(AbstractDatasetHelper):
            ...
            @staticmethod
            def next_batch_task(batch):
                # implement next_batch_task for every iteration of mini batch
                # if no need to implement next_batch_task, just return batch
                x, label = batch[0], batch[1]
                return x, label
        ```


    4. implement self.load_dataset()

        * must return dataset and input_shapes

        ```python
        ...
        class ExampleDatasetHelper(AbstractDatasetHelper):
            ...
            @staticmethod
            def load_dataset(limit=None):
                # implement helper for load_dataset
                # must return loaded dataset and dataset's input_shapes
                # input_shapes is dict.
                # key is input_shape_key in dict_keys.input_shape_keys
                # value numpy shape of data
                example_data = ExampleDataset(preprocess=ExampleDatasetHelper.preprocess)
                example_data.load(EXAMPLE_DATASET_PATH, limit=limit)
                input_shapes = {
                    INPUT_SHAPE_KEY_DATA_X: [32, 32, 1],
                    INPUT_SHAPE_KEY_LABEL: [10],
                    INPUT_SHAPE_KEY_LABEL_SIZE: 10,
                }

                return example_data, input_shapes
         ```




