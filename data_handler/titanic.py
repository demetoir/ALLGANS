from data_handler.AbstractDataset import *
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
from util.numpy_utils import *
import pandas as pd
from data_handler.BaseDataset import BaseDataset, DatasetCollection


def np_str_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    new_arr = np.zeros_like(np_arr)
    for idx, label in enumerate(labels):
        new_arr = np.where(np_arr == label, idx, new_arr)

    return new_arr.astype(dtype=np.int)


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


def x_preprocess(self):
    data = self.data[SEX]
    self.data["Sex_male"] = np.where(data == "male", 1, 0).reshape([-1, 1])
    self.data["Sex_female"] = np.where(data == "female", 1, 0).reshape([-1, 1])

    data = self.data[EMBARKED]
    self.data["Embarked_C"] = np.where(data == "C", 1, 0).reshape([-1, 1])
    self.data["Embarked_S"] = np.where(data == "S", 1, 0).reshape([-1, 1])
    self.data["Embarked_Q"] = np.where(data == "Q", 1, 0).reshape([-1, 1])
    self.data["Embarked_nan"] = np.where(data == "nan", 1, 0).reshape([-1, 1])

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

    data = self.data[AGE]

    a = np.zeros(list(data.shape) + [2])
    for i in range(len(data)):
        if data[i] == "nan":
            a[i] = [0, 1]
        else:
            a[i] = [float(data[i]) / 100, 0]
    self.data[AGE] = a

    data = self.data[FARE]
    a = np.zeros(list(data.shape) + [2])
    for i in range(len(data)):
        if data[i] == "nan":
            a[i] = [0, 1]
        else:
            a[i] = [float(data[i]) / 600, 0]
    self.data[FARE] = a

    data = self.get_datas([
        "Sex_male",
        "Sex_female",
        "Embarked_C",
        "Embarked_S",
        "Embarked_Q",
        "Embarked_nan",
        PCLASS,
        SIBSP,
        PARCH,
        AGE,
        FARE,
    ])
    return np.concatenate(data, axis=1)


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
        pd_data = pd_data.fillna("nan")
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

    def save(self):
        pass

    def preprocess(self):
        data = x_preprocess(self)
        self.add_data(BK_X, data)

        # add train_label
        data = self.data[SURVIVED]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.data[SURVIVED] = data

        data = self.get_datas([
            SURVIVED
        ])
        self.add_data(BK_LABEL, data[0])


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
        pd_data = pd_data.fillna('nan')
        for col, key in zip(self.CSV_COLUMNS, self.BATCH_KEYS):
            self.data[key] = np.array(pd_data[col])[1:]

    def save(self):
        pass

    def preprocess(self):
        data = x_preprocess(self)
        self.add_data(BK_X, data)


class titanic(DatasetCollection):
    LABEL_SIZE = 2

    def __init__(self):
        super().__init__()
        self.train_set = titanic_train()
        self.test_set = titanic_test()
        self.validation_set = None

    def load(self, path, **kwargs):
        super().load(path, **kwargs)
        self.train_set.shuffle()
        ratio = (8, 2)
        self.train_set, self.validation_set = self.train_set.split(ratio=ratio)
        print("split train set to train and validation set ratio=%s" % str(ratio))
