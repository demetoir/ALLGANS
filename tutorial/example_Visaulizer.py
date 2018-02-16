from visualizer.AbstractVisualizer import AbstractVisualizer


class user_define_visualizer(AbstractVisualizer):
    """visualize a tile image from GAN's result images"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        # self.visualizer_path

        data = sess.run(model.G, feed_dict={model.z: model.get_noise()})



