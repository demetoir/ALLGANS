from sklearn_like_toolkit.Base import BaseClass


class BaseSklearn(BaseClass):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, Xs, Ys, Ys_type=None):
        pass

    def predict(self, Xs):
        pass

    def score(self, Xs, Ys, Ys_type):
        pass

    def proba(self, Xs, transpose_shape=True):
        pass
