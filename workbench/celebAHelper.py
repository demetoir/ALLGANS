# CelebA dataset helper
from data_handler.celebA import celebA
from env_settting import CELEBA_PATH


class CelebAHelper:
    # todo implement this class
    @staticmethod
    def preprocess(dataset):
        pass

    @staticmethod
    def batch_after_task(batch):
        x = batch[0]
        return x

    @staticmethod
    def load_dataset():
        dataset = celebA(preprocess=CelebAHelper.preprocess, batch_after_task=CelebAHelper.batch_after_task)
        dataset.build_pkl(CELEBA_PATH)
        # dataset.load(CELEBA_PATH, )
        input_shape = [32, 32, 3]
        return dataset, input_shape
