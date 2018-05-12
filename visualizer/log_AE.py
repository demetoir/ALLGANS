from visualizer.AbstractVisualizer import AbstractVisualizer


class log_AE(AbstractVisualizer):
    """visualizer error of discriminator AE in BEGAN"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        AE_loss_mean, global_step = model.run(
            sess,
            [model.loss_mean, model.global_step],
            dataset
        )

        self.log(
            'global_step : %04d ' % global_step,
            'AE loss: {:.4} '.format(AE_loss_mean),
        )
