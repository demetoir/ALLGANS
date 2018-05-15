from visualizer.AbstractVisualizer import AbstractVisualizer


class log_GAN_loss(AbstractVisualizer):
    """visualize log loss of GAN"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        loss_D, loss_G, global_step = model.run(
            sess,
            [model.loss_D, model.loss_G, model.global_step],
            dataset
        )

        self.log('global_step : %04d ' % global_step
                 + 'D loss: {:.4} '.format(loss_D)
                 + 'G loss: {:.4} '.format(loss_G))
