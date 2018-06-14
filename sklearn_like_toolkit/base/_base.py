from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score

from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT

CLF_METRICS = {
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'roc_auc_score': roc_auc_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
}


class _Reformat_Ys:
    def _reformat_to_index(self, Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_INDEX)

    def _reformat_to_onehot(self, Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_ONEHOT)


class _clf_metric:
    def __init__(self):
        self._metrics = CLF_METRICS

    def _apply_metric(self, Y_true, Y_predict, metric):
        return self._metrics[metric](Y_true, Y_predict)

    def _apply_metric_pack(self, Y_true, Y_predict):
        return {
            key: self._apply_metric(Y_true, Y_predict, key)
            for key in self._metrics
        }