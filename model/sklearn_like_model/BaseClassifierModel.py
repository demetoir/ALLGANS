from data_handler.BaseDataset import BaseDataset
from model.sklearn_like_model.BaseModel import BaseModel


class Dataset(BaseDataset):
    def load(self, path, limit=None):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass


class BaseClassifierModel(BaseModel):

    def build_input_shapes(self, input_shapes):
        raise NotImplementedError

    def build_main_graph(self):
        raise NotImplementedError

    def build_loss_function(self):
        raise NotImplementedError

    def build_train_ops(self):
        raise NotImplementedError

    @property
    def _Xs(self):
        raise NotImplementedError

    @property
    def _Ys(self):
        raise NotImplementedError

    @property
    def _train_ops(self):
        raise NotImplementedError

    @property
    def _predict_ops(self):
        raise NotImplementedError

    @property
    def _score_ops(self):
        raise NotImplementedError

    @property
    def _proba_ops(self):
        raise NotImplementedError

    @property
    def _metric_ops(self):
        raise NotImplementedError

    def train(self, Xs, Ys, epoch=100, save_interval=None, batch_size=None):
        self.if_not_ready_to_train()

        dataset = Dataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size
        for e in range(epoch):
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

            Xs, Ys = dataset.next_batch(batch_size, batch_keys=['Xs', 'Ys'], look_up=False)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
            self.log.info("e:{}, i:{} loss : {}".format(e, iter_num, loss))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def predict(self, Xs):
        return self.sess.run(self._predict_ops, feed_dict={self._Xs: Xs})

    def score(self, Xs, Ys):
        return self.sess.run(self._score_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})

    def proba(self, Xs):
        return self.sess.run(self._proba_ops, feed_dict={self._Xs: Xs})

    def metric(self, Xs, Ys):
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs, self._Ys: Ys})
