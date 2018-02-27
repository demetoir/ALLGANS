from data_handler.AbstractDataset import *
import pandas as pd
import numpy as np


# from env_settting import TITANIC_PATH


class titanic(AbstractDataset):
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

    TRAIN_PASSENGERID = 'TRAIN_PASSENGERID'
    TRAIN_SURVIVED = 'TRAIN_SURVIVED'
    TRAIN_PCLASS = 'TRAIN_PCLASS'
    TRAIN_NAME = 'TRAIN_NAME'
    TRAIN_SEX = 'TRAIN_SEX'
    TRAIN_AGE = 'TRAIN_AGE'
    TRAIN_SIBSP = 'TRAIN_SIBSP'
    TRAIN_PARCH = 'TRAIN_PARCH'
    TRAIN_TICKET = 'TRAIN_TICKET'
    TRAIN_FARE = 'TRAIN_FARE'
    TRAIN_CABIN = 'TRAIN_CABIN'
    TRAIN_EMBARKED = 'TRAIN_EMBARKED'

    TRAIN_BATCH_KEYS = [
        TRAIN_PASSENGERID,
        TRAIN_SURVIVED,
        TRAIN_PCLASS,
        TRAIN_NAME,
        TRAIN_SEX,
        TRAIN_AGE,
        TRAIN_SIBSP,
        TRAIN_PARCH,
        TRAIN_TICKET,
        TRAIN_FARE,
        TRAIN_CABIN,
        TRAIN_EMBARKED,
    ]

    TEST_PASSENGERID = 'TEST_PASSENGERID'
    TEST_SURVIVED = 'TEST_SURVIVED'
    TEST_PCLASS = 'TEST_PCLASS'
    TEST_NAME = 'TEST_NAME'
    TEST_SEX = 'TEST_SEX'
    TEST_AGE = 'TEST_AGE'
    TEST_SIBSP = 'TEST_SIBSP'
    TEST_PARCH = 'TEST_PARCH'
    TEST_TICKET = 'TEST_TICKET'
    TEST_FARE = 'TEST_FARE'
    TEST_CABIN = 'TEST_CABIN'
    TEST_EMBARKED = 'TEST_EMBARKED'
    TEST_BATCH_KEYS = [
        TEST_PASSENGERID,
        TEST_PCLASS,
        TEST_NAME,
        TEST_SEX,
        TEST_AGE,
        TEST_SIBSP,
        TEST_PARCH,
        TEST_TICKET,
        TEST_FARE,
        TEST_CABIN,
        TEST_EMBARKED,
    ]

    TEST_CSV = "test.csv"
    TRAIN_CSV = "train.csv"

    def __init__(self, preprocess=None, batch_after_task=None, before_load_task=None):
        super().__init__(preprocess, batch_after_task, before_load_task)

        self.batch_keys = self.TRAIN_BATCH_KEYS + self.TEST_BATCH_KEYS

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
            names=self.TRAIN_CSV_COLUMNs
        )
        for col, key in zip(self.TRAIN_CSV_COLUMNs, self.TRAIN_BATCH_KEYS):
            self.data[key] = np.array(train_data[col])[1:]

        test_data_path = os.path.join(path, self.TEST_CSV)
        test_data = pd.read_csv(
            test_data_path,
            sep=',',
            header=None,
            error_bad_lines=False,
            names=self.TEST_CSV_COLUMNs
        )
        for col, key in zip(self.TEST_CSV_COLUMNs, self.TEST_BATCH_KEYS):
            self.data[key] = np.array(test_data[col])[1:]

    def save(self):
        pass


class titanicHelper(AbstractDatasetHelper):
    @staticmethod
    def load_dataset(path, limit=None):
        dataset = titanic(
            preprocess=titanicHelper.preprocess,
            batch_after_task=titanicHelper.next_batch_task
        )
        dataset.load(path=path, limit=limit)
        input_shapes = {
            titanic.TRAIN_PASSENGERID: [1],
            titanic.TRAIN_SURVIVED: [1],
            titanic.TRAIN_PCLASS: [1],
            titanic.TRAIN_NAME: [1],
            titanic.TRAIN_SEX: [1],
            titanic.TRAIN_AGE: [1],
            titanic.TRAIN_SIBSP: [1],
            titanic.TRAIN_PARCH: [1],
            titanic.TRAIN_TICKET: [1],
            titanic.TRAIN_FARE: [1],
            titanic.TRAIN_CABIN: [1],
            titanic.TRAIN_EMBARKED: [1],

            titanic.TEST_PASSENGERID: [1],
            titanic.TEST_PCLASS: [1],
            titanic.TEST_NAME: [1],
            titanic.TEST_SEX: [1],
            titanic.TEST_AGE: [1],
            titanic.TEST_SIBSP: [1],
            titanic.TEST_PARCH: [1],
            titanic.TEST_TICKET: [1],
            titanic.TEST_FARE: [1],
            titanic.TEST_CABIN: [1],
            titanic.TEST_EMBARKED: [1],
        }

        return dataset, input_shapes

    @staticmethod
    def next_batch_task(batch):
        return super().next_batch_task(batch)

    @staticmethod
    def preprocess(dataset):
        pass
