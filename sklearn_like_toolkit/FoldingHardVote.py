import numpy as np
from data_handler.DummyDataset import DummyDataset
from sklearn_like_toolkit.base._base import _clf_metric, _Reformat_Ys


class FoldingHardVote(_Reformat_Ys, _clf_metric):
    def __init__(self, clfs, split_rate=0.8):
        super().__init__()
        self.clfs = [self._clone(clf) for clf in clfs]
        self.n = len(self.clfs)
        self.class_size = None
        self._is_fitted = False
        self.split_rate = split_rate

    def _clone(self, clf):
        return clf.__class__()

    def _collect_predict(self, Xs):
        return np.array([clf.predict(Xs) for clf in self.clfs])

    def fit(self, Xs, Ys):
        self.class_size = self._reformat_to_onehot(Ys).shape[1]
        dset = DummyDataset({'Xs': Xs, 'Ys': Ys})

        for clf in self.clfs:
            dset.shuffle()
            Xs, Ys = dset.next_batch(int(dset.size * self.split_rate))
            clf.fit(Xs, Ys)

        self._is_fitted = True

    def predict_bincount(self, Xs):
        predicts = self._collect_predict(Xs).transpose()
        bincount = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.class_size),
            axis=1, arr=predicts
        )
        return bincount

    def predict_proba(self, Xs):
        return self.predict_bincount(Xs) / float(self.n)

    def predict(self, Xs):
        predicts = self._collect_predict(Xs).transpose()
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=self.class_size)),
            axis=1, arr=predicts)
        return maj

    def score(self, Xs, Ys, metric='accuracy'):
        Ys = self._reformat_to_index(Ys)
        return self._apply_metric(Ys, self.predict(Xs), metric)

    def score_pack(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
        return self._apply_metric_pack(Ys, self.predict(Xs))
