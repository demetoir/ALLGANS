from data_handler.AbstractDataset import *
from dict_keys.input_shape_keys import *
from util.numpy_utils import *
import pandas as pd
from data_handler.BaseDataset import BaseDataset, DatasetCollection


def np_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    for idx, label in enumerate(labels):
        np_arr[np_arr == label] = np.array([idx], dtype=np.int32)
    return np_arr.astype(dtype=np.int32)


FOLDER_NAME = "tatanic"
PASSENGERID = 'PassengerId'
SURVIVED = 'Survived'
PCLASS = 'Pclass'
NAME = 'Name'
SEX = 'Sex'
AGE = 'Age'
SIBSP = 'SibSp'
PARCH = 'Parch'
TICKET = 'Ticket'
FARE = 'Fare'
CABIN = 'Cabin'
EMBARKED = 'Embarked'


class titanic_train(BaseDataset):
    BATCH_KEYS = [
        PASSENGERID,
        SURVIVED,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    CSV_COLUMNS = [
        PASSENGERID,
        SURVIVED,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    FILE_NAME = "train.csv"

    def load(self, path, limit=None):
        data_path = os.path.join(path, self.FILE_NAME)
        pd_data = pd.read_csv(
            data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.CSV_COLUMNS,
        )
        pd_data = pd_data.fillna("None")
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

    def save(self):
        pass

    def preprocess(self):
        data = self.data[SURVIVED]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[SURVIVED] = data

        data = self.data[SEX]
        labels = ["male", "female"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.data[SEX] = data

        data = self.data[EMBARKED]
        labels = ["C", "S", "Q", "None"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.data[EMBARKED] = data

        data = self.data[PCLASS]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[PCLASS] = data

        data = self.data[SIBSP]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[SIBSP] = data

        data = self.data[PARCH]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[PARCH] = data

        # add train_x
        data = self.get_datas([
            SEX,
            EMBARKED,
            PCLASS,
            SIBSP,
            PARCH
        ])
        data = np.concatenate(data, axis=1)
        self.add_data(ISK_TRAIN_X, data)

        # add train_label
        data = self.get_datas([
            SURVIVED
        ])
        self.add_data(ISK_TRAIN_LABEL, data[0])


class titanic_test(BaseDataset):
    BATCH_KEYS = [
        PASSENGERID,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    CSV_COLUMNS = [
        PASSENGERID,
        PCLASS,
        NAME,
        SEX,
        AGE,
        SIBSP,
        PARCH,
        TICKET,
        FARE,
        CABIN,
        EMBARKED,
    ]

    FILE_NAME = "test.csv"

    def load(self, path, limit=None):
        data_path = os.path.join(path, self.FILE_NAME)
        pd_data = pd.read_csv(
            data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.CSV_COLUMNS,
        )
        pd_data = pd_data.fillna("None")
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

        pass

    def save(self):
        pass

    def preprocess(self):
        data = self.data[SEX]
        labels = ["male", "female"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.data[SEX] = data

        data = self.data[EMBARKED]
        labels = ["C", "S", "Q", "None"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.data[EMBARKED] = data

        data = self.data[PCLASS]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[PCLASS] = data

        data = self.data[SIBSP]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[SIBSP] = data

        data = self.data[PARCH]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[PARCH] = data

        data = self.get_datas([
            SEX,
            EMBARKED,
            PCLASS,
            SIBSP,
            PARCH
        ])
        data = np.concatenate(data, axis=1)
        self.add_data(ISK_TEST_X, data)


class titanic(DatasetCollection):

    def __init__(self):
        super().__init__()
        self.train_set = titanic_train()
        self.test_set = titanic_test()
        self.validation_set = None

    def load(self, path, **kwargs):
        super().load(path, **kwargs)
        ratio = (6, 4)
        self.train_set, self.validation_set = self.train_set.split(ratio=ratio)

    @property
    def input_shapes(self):
        return self.train_set.input_shapes
