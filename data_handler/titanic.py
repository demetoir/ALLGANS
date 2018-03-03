from data_handler.AbstractDataset import *
from dict_keys.input_shape_keys import *
from util.numpy_utils import *
import matplotlib.pyplot as plt
import pandas as pd
from data_handler.BaseDataset import BaseDataset
import sklearn


# from env_settting import TITANIC_PATH


class titanic(BaseDataset):
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

    TEST_CSV_COLUMNs = [
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
    TRAIN_CSV_COLUMNs = [
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
    TRAIN_BATCH_KEYS = [
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
    TEST_BATCH_KEYS = [
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

    # TRAIN_PASSENGERID = 'TRAIN_PASSENGERID'
    # TRAIN_SURVIVED = 'TRAIN_SURVIVED'
    # TRAIN_PCLASS = 'TRAIN_PCLASS'
    # TRAIN_NAME = 'TRAIN_NAME'
    # TRAIN_SEX = 'TRAIN_SEX'
    # TRAIN_AGE = 'TRAIN_AGE'
    # TRAIN_SIBSP = 'TRAIN_SIBSP'
    # TRAIN_PARCH = 'TRAIN_PARCH'
    # TRAIN_TICKET = 'TRAIN_TICKET'
    # TRAIN_FARE = 'TRAIN_FARE'
    # TRAIN_CABIN = 'TRAIN_CABIN'
    # TRAIN_EMBARKED = 'TRAIN_EMBARKED'
    # TRAIN_BATCH_KEYS = [
    #     TRAIN_PASSENGERID,
    #     TRAIN_SURVIVED,
    #     TRAIN_PCLASS,
    #     TRAIN_NAME,
    #     TRAIN_SEX,
    #     TRAIN_AGE,
    #     TRAIN_SIBSP,
    #     TRAIN_PARCH,
    #     TRAIN_TICKET,
    #     TRAIN_FARE,
    #     TRAIN_CABIN,
    #     TRAIN_EMBARKED,
    # ]
    #
    # TEST_PASSENGERID = 'TEST_PASSENGERID'
    # TEST_PCLASS = 'TEST_PCLASS'
    # TEST_NAME = 'TEST_NAME'
    # TEST_SEX = 'TEST_SEX'
    # TEST_AGE = 'TEST_AGE'
    # TEST_SIBSP = 'TEST_SIBSP'
    # TEST_PARCH = 'TEST_PARCH'
    # TEST_TICKET = 'TEST_TICKET'
    # TEST_FARE = 'TEST_FARE'
    # TEST_CABIN = 'TEST_CABIN'
    # TEST_EMBARKED = 'TEST_EMBARKED'
    # TEST_BATCH_KEYS = [
    #     TEST_PASSENGERID,
    #     TEST_PCLASS,
    #     TEST_NAME,
    #     TEST_SEX,
    #     TEST_AGE,
    #     TEST_SIBSP,
    #     TEST_PARCH,
    #     TEST_TICKET,
    #     TEST_FARE,
    #     TEST_CABIN,
    #     TEST_EMBARKED,
    # ]

    TEST_CSV = "test.csv"
    TRAIN_CSV = "train.csv"

    def __init__(self):
        super().__init__()

        self.batch_keys = self.BATCH_KEYS
        # self.download_infos = [
        #     DownloadInfo(
        #         url='https://www.kaggle.com/c/3136/download/gender_submission.csv',
        #         is_zipped=False,
        #         download_file_name="gender_submission.csv",
        #     ),
        #     DownloadInfo(
        #         url='https://www.kaggle.com/c/3136/download/test.csv',
        #         is_zipped=False,
        #         download_file_name=self.TEST_CSV,
        #     ),
        #     DownloadInfo(
        #         url='https://www.kaggle.com/c/3136/download/train.csv',
        #         is_zipped=False,
        #         download_file_name=self.TRAIN_CSV,
        #     )
        # ]

    def load(self, path, limit=None):
        train_data_path = os.path.join(path, self.TRAIN_CSV)
        train_data = pd.read_csv(
            train_data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.TRAIN_CSV_COLUMNs,
        )
        train_data = train_data.fillna("None")
        for col, key in zip(self.TRAIN_CSV_COLUMNs, self.TRAIN_BATCH_KEYS):
            self.train_data[key] = np.array(train_data[col])[1:]

        test_data_path = os.path.join(path, self.TEST_CSV)
        test_data = pd.read_csv(
            test_data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.TEST_CSV_COLUMNs,
        )
        test_data = test_data.fillna("None")
        for col, key in zip(self.TEST_CSV_COLUMNs, self.TEST_BATCH_KEYS):
            self.test_data[key] = np.array(test_data[col])[1:]

    def save(self):
        pass

    def preprocess(self):
        # train data
        data = self.train_data[titanic.SURVIVED]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.train_data[titanic.SURVIVED] = data

        data = self.train_data[titanic.SEX]
        labels = ["male", "female"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.train_data[titanic.SEX] = data

        data = self.train_data[titanic.EMBARKED]
        labels = ["C", "S", "Q", "None"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.train_data[titanic.EMBARKED] = data

        data = self.train_data[titanic.PCLASS]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.train_data[titanic.PCLASS] = data

        data = self.train_data[titanic.SIBSP]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.train_data[titanic.SIBSP] = data

        data = self.train_data[titanic.PARCH]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.train_data[titanic.PARCH] = data

        # with update_dict_value(dataset, titanic.TRAIN_AGE) as data:
        #
        #     print(data)

        # test data
        data = self.test_data[titanic.SEX]
        labels = ["male", "female"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.train_data[titanic.SEX] = data

        data = self.test_data[titanic.EMBARKED]
        labels = ["C", "S", "Q", "None"]
        data = np_labels_to_index(data, labels)
        data = np_index_to_onehot(data)
        self.test_data[titanic.EMBARKED] = data

        data = self.test_data[titanic.PCLASS]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.test_data[titanic.PCLASS] = data

        data = self.test_data[titanic.SIBSP]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.test_data[titanic.SIBSP] = data

        data = self.test_data[titanic.PARCH]
        data = data.astype(np.int)
        data = np_index_to_onehot(data)
        self.test_data[titanic.PARCH] = data

        # with update_dict_value(dataset, titanic.TRAIN_FARE) as data:
        #     data = data.astype(np.float)
        #     # bin_ = [0, 4, 6, 8, 10, 20, 100, 1000]
        #     # plt.hist(data, bins=bin_)
        #
        #     plt.show()

        # add train_x
        data = self.get_data([
            titanic.TRAIN_SEX,
            titanic.TRAIN_EMBARKED,
            titanic.TRAIN_PCLASS,
            titanic.TRAIN_SIBSP,
            titanic.TRAIN_PARCH
        ])
        data = np.concatenate(data, axis=1)
        self.add_data(ISK_TRAIN_X, data)

        # add test_x
        data = self.get_data([
            titanic.TEST_SEX,
            titanic.TEST_EMBARKED,
            titanic.TEST_PCLASS,
            titanic.TEST_SIBSP,
            titanic.TEST_PARCH
        ])
        data = np.concatenate(data, axis=1)
        self.add_data(ISK_TEST_X, data)

        # add train_label
        data = self.get_data([
            titanic.TRAIN_SURVIVED
        ])
        self.add_data(ISK_TRAIN_LABEL, data)


def np_labels_to_index(np_arr, labels):
    np_arr = np.asarray(np_arr)
    for idx, label in enumerate(labels):
        np_arr[np_arr == label] = np.array([idx], dtype=np.int32)
    return np_arr.astype(dtype=np.int32)
