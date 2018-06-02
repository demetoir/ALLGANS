# usage
# after load instance to InstanceManager
from visualizer.VisualizerClassLoader import VisualizerClassLoader

visualizer = VisualizerClassLoader.load("visualizer_name")
manager.load_visualizer(visualizer, execute_interval=10)


## Implement step
#1. define visualizer class and inherit AbstractVisualizer
from visualizer.AbstractVisualizer import AbstractVisualizer

class example_Visualizer(AbstractVisualizer):
#2. implement **task()**
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        # fetch data by sess.run()
        sample_imgs0 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs1 = sess.run(model.G, feed_dict={model.z: model.get_noise()})
        sample_imgs = np.concatenate((sample_imgs0, sample_imgs1))
        sample_imgs = np_img_float32_to_uint8(sample_imgs)

        # use self.visualizer_path for save result
        img_path = os.path.join(self.visualizer_path, '{}.png'.format(str(iter_num).zfill(5)))
        tile = np_img_to_tile(sample_imgs, column_size=8)
        pil_img = np_img_to_PIL_img(tile)
        with open(img_path, 'wb') as fp:
            pil_img.save(fp)
