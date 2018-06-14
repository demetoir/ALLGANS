from data_handler.BaseDataset import BaseDataset


class DummyDataset(BaseDataset):
    def load(self, path, limit=None):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass

