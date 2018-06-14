from model.sklearn_like_model.BaseModel import BaseModel
import numpy as np
from model.sklearn_like_model.DummyDataset import DummyDataset


class BaseAutoEncoder(BaseModel):
    VERSION = 1.0

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
    def _zs(self):
        raise NotImplementedError

    @property
    def _train_ops(self):
        raise NotImplementedError

    @property
    def _code_ops(self):
        raise NotImplementedError

    @property
    def _recon_ops(self):
        raise NotImplementedError

    @property
    def _generate_ops(self):
        raise NotImplementedError

    @property
    def _metric_ops(self):
        raise NotImplementedError

    def train(self, Xs, epoch=100, save_interval=None, batch_size=None):
        self.if_not_ready_to_train()

        dataset = DummyDataset()
        dataset.add_data('Xs', Xs)

        if batch_size is None:
            batch_size = self.batch_size

        iter_num = 0
        iter_per_epoch = dataset.size // batch_size
        self.log.info("train epoch {}, iter/epoch {}".format(epoch, iter_per_epoch))

        for e in range(epoch):
            dataset.shuffle()
            for i in range(iter_per_epoch):
                iter_num += 1

                Xs = dataset.next_batch(batch_size, batch_keys=['Xs'])
                self.sess.run(self._train_ops, feed_dict={self._Xs: Xs})

            Xs = dataset.next_batch(batch_size, batch_keys=['Xs'], look_up=False)
            loss = self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs})
            loss = np.mean(loss)
            self.log.info("e:{e} loss : {loss}".format(e=e, loss=loss))

            if save_interval is not None and e % save_interval == 0:
                self.save()

    def code(self, Xs):
        return self.sess.run(self._code_ops, feed_dict={self._Xs: Xs})

    def recon(self, Xs):
        return self.sess.run(self._recon_ops, feed_dict={self._Xs: Xs})

    def generate(self, zs):
        return self.sess.run(self._recon_ops, feed_dict={self._zs: zs})

    def random_z(self):
        raise NotImplementedError

    def metric(self, Xs):
        return self.sess.run(self._metric_ops, feed_dict={self._Xs: Xs})
