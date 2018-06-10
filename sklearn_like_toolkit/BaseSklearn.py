from sklearn_like_toolkit.Base import BaseClass


class BaseSklearn(BaseClass):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, Xs, Ys, **kwargs):
        pass

    def predict(self, Xs, **kwargs):
        pass

    def score(self, Xs, Ys, Ys_type, **kwargs):
        pass

    def proba(self, Xs, **kwargs):
        pass
