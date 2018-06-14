from visualizer.AbstractVisualizer import AbstractVisualizer


class log_AAE(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        ret = model.run(
            sess,
            [model.loss_AE_mean,
             model.loss_D_gauss_mean, model.loss_D_cate_mean,
             model.loss_G_gauss_mean, model.loss_G_cate_mean,
             model.acc, model.global_step],
            dataset
        )
        AE_loss_mean, loss_D_gauss_mean, loss_D_cate_mean, loss_G_gauss_mean, loss_G_cate_mean, acc, global_step = ret
        self.log(
            'global_step : %04d ' % global_step,
            'AE loss: {:.4} '.format(AE_loss_mean),
            'D gauss loss: {:.4} '.format(loss_D_gauss_mean),
            'D cate loss: {:.4} '.format(loss_D_cate_mean),
            'G gauss loss: {:.4} '.format(loss_G_gauss_mean),
            'G cate loss: {:.4} '.format(loss_G_cate_mean),
            'acc: {:.4} '.format(acc),
        )
