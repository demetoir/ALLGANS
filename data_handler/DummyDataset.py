from data_handler.BaseDataset import DatasetCollection, BaseDataset


class DummyDataset(BaseDataset):
    def __init__(self, set=None):
        super().__init__()

        if set is not None:
            for key in set:
                self.add_data(key, set[key])
