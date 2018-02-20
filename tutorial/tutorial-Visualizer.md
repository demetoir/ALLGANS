# Document - Visualizer

## Details
Visualizer class is visualize current training instance or sampling instance


## usage

* execute_interval: interval to execute visualizer

```python
# after load instance to InstanceManager
from VisualizerClassLoader import VisualizerClassLoader

visualizer = VisualizerClassLoader.load_class("visualizer_name")
manager.load_visualizer(visualizer, execute_interval=10)
```


## Implement step

1. define visualizer class and inherit AbstractVisualizer
    * make .py file in Visualizer folder.
    * .py file name and class name must same

    ```python
    from visualizer.AbstractVisualizer import AbstractVisualizer

    class example_Visualizer(AbstractVisualizer):
        pass

    ```
            """visualize a tile image from GAN's result images"""


2. implement **task()**

    * self.visualizer_path : directory path of visualizer's output file in current instance

    argument sess, iter_num, model, dataset are feed from InstanceManager

    * sess : tensorflow.Session object
    * iter_num : current iteration number
    * model : current training or sampling model
    * dataset : current training or sampling dataset

    ```python
    ...
    class example_Visualizer(AbstractVisualizer):
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
    ```