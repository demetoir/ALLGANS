import numpy as np
from sklearn.metrics import accuracy_score
from data_handler.DummyDataset import DummyDataset
from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_ONEHOT, NP_ARRAY_TYPE_INDEX


class FoldingHardVote:
    def __init__(self, clfs, split_rate=0.8):
        self.clfs = [self._clone(clf) for clf in clfs]
        self.n = len(self.clfs)
        self.class_size = None
        self._is_fitted = False
        self.split_rate = split_rate

    def _clone(self, clf):
        return clf.__class__()

    def _check_is_fitted(self):
        return self._is_fitted

    def _collect_predict(self, Xs):
        return np.array([clf.predict(Xs) for clf in self.clfs])

    def _collect_proba(self, Xs):
        pass

    def fit(self, Xs, Ys):
        self.class_size = reformat_np_arr(Ys, NP_ARRAY_TYPE_ONEHOT).shape[1]
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

    def score(self, Xs, Ys, sample_weight=None):
        Ys = reformat_np_arr(Ys, NP_ARRAY_TYPE_INDEX)
        return accuracy_score(Ys, self.predict(Xs), sample_weight=sample_weight)
