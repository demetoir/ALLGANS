from env_settting import ROOT_PATH
from unit_test.DummyModel import DummyModel


class test_AbstractModel:
    def __init__(self):
        pass

    def test__00(self):
        shape = (32, 32, 3)
        path = ROOT_PATH
        model = DummyModel(path)
