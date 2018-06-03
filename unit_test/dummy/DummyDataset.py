class DummyDataset:
    def __init__(self):
        self.data_size = 2

    @staticmethod
    def next_batch(batch_size):
        print('dummy dataset next_batch size = %d' % batch_size)
