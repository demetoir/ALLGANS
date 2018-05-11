from data_handler.AbstractDataset import *
from dict_keys.dataset_batch_keys import *
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
    x_dict = {}

    data = self.data[SEX]
    x_dict["Sex_male"] = np.where(data == "male", 1, 0).reshape([-1, 1])
    x_dict["Sex_female"] = np.where(data == "female", 1, 0).reshape([-1, 1])

    data = self.data[EMBARKED]
    x_dict["Embarked_C"] = np.where(data == "C", 1, 0).reshape([-1, 1])
    x_dict["Embarked_S"] = np.where(data == "S", 1, 0).reshape([-1, 1])
    x_dict["Embarked_Q"] = np.where(data == "Q", 1, 0).reshape([-1, 1])
    x_dict["Embarked_nan"] = np.where(data == "nan", 1, 0).reshape([-1, 1])

    data = self.data[PCLASS]
    data = data.astype(np.int)
    data = np_index_to_onehot(data)
    x_dict["pclass_onehot"] = data

    data = self.data[SIBSP]
    data = data.astype(np.int)
    data = np_index_to_onehot(data)
    x_dict["sibsp_onehot"] = data

    data = self.data[PARCH]
    data = data.astype(np.int)
    data = np_index_to_onehot(data, n=10)
    x_dict["parch_onehot"] = data

    sibsp = self.data[SIBSP].astype(np.int)
    parch = self.data[PARCH].astype(np.int)
    x_dict["family_size_onehot"] = np_index_to_onehot(sibsp + parch + 1, n=20)

    data = self.data[AGE]
    a = np.zeros(list(data.shape) + [2])
    for i in range(len(data)):
        if data[i] == "nan":
            a[i] = [0, 1]
        else:
            a[i] = [float(data[i]) / 100, 0]
    x_dict["age_scaled"] = a

    data = self.data[FARE]
    a = np.zeros(list(data.shape) + [2])
    for i in range(len(data)):
        if data[i] == "nan":
            a[i] = [0, 1]
        else:
            a[i] = [float(data[i]) / 600, 0]
    x_dict["fare_scaled_"] = a

    return np.concatenate(list(x_dict.values()), axis=1)


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
        ratio = (7, 3)
        self.train_set, self.validation_set = self.train_set.split(ratio=ratio)
        self.log("split train set to train and validation set ratio=%s" % str(ratio))
