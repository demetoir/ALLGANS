class BaseClass:
    def __str__(self):
        return self.__class__.__name__

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass


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
