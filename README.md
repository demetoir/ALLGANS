# ALLGANs

머신러닝 프레임워크를 활용한 비교사(Unsupervised) 학습 모델 구현 프로젝트

## Index

- [ALLGANs](#allgans)
  - [Training dataset](#training-dataset)
  - [Results](#results)
    - [GAN](#gan)
    - [LSGAN](#lsgan)
    - [WGAN](#wgan)
  - [observation](#observation)
  - [Getting Started](#getting-started)
  - [Folder structure](#folder-structure)
  - [Dependency](#dependency)
  - [Authors](#authors)
  - [Reference](#reference)

## Training Dataset

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [LLD](https://data.vision.ee.ethz.ch/cvl/lld/)(Large Logo Dataset)

## Results

- GAN, LSGAN, WGAN are have same generator and discriminator CNN structure.
- no data augmentation.

### GAN

|   dataset    |             MNIST(20 epoch)              |         fashion-mnist(20 epoch)          |
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image |    ![GAN_MNIST_img](./result_images/GAN-MNIST.png)    | ![GAN_Fashion-MNIST_img](./result_images/GAN-fashion_mnist.png) |
|    loss D    | ![GAN_MNIST_loss-d](./result_images/GAN-MNIST-loss_D.png ) | ![GAN_Fashion-MNIST_loss-d](./result_images/GAN-fashion-mnist-loss_D.png) |
| loss D_real  | ![GAN_MNIST_loss-d-real](./result_images/GAN-MNIST-loss_D_real.png ) | ![GAN_Fashion-MNIST_loss-d-real](./result_images/GAN-fashion-mnist-loss_D_real.png) |
|  loss D_gen  | ![GAN_MNIST_loss-d-gen](./result_images/GAN-MNIST-loss_D_gen.png ) | ![GAN_Fashion-MNIST_loss-d-gen](./result_images/GAN-fashion-mnist-loss_D_gen.png) |
|    loss G    | ![GAN_MNIST_loss-g](./result_images/GAN-MNIST-loss_G.png ) | ![GAN_Fashion-MNIST_loss-g](./result_images/GAN-fashion-mnist-loss_G.png) |

|   dataset    |            CIFAR10(20 epoch)             |              LLD(20 epoch)               |
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image |   ![GAN_CIFAR10_img](./result_images/GAN-CIFAR10.png)   |     ![GAN_LLD_img](./result_images/GAN-LLD.png)     |
|    loss D    | ![GAN_CIFAR10_loss-d](./result_images/GAN-CIFAR10-loss_D.png) | ![GAN_LLD_loss-d](./result_images/GAN-LLD-loss_D.png)  |
| loss D_real  | ![GAN_CIFAR10_loss-d-real](./result_images/GAN-CIFAR10-loss_D_real.png) | ![GAN_LLD_loss-d-real](./result_images/GAN-LLD-loss_D_real.png) |
|  loss D_gen  | ![GAN_CIFAR10_loss-d-gen](./result_images/GAN-CIFAR10-loss_D_gen.png ) | ![GAN_LLD_loss-d-gen](./result_images/GAN-LLD-loss_D_gen.png) |
|    loss G    | ![GAN_CIFAR10_loss-g](./result_images/GAN-CIFAR10-loss_G.png ) | ![GAN_LLD_loss-g](./result_images/GAN-LLD-loss_G.png)  |

### LSGAN

|   dataset    |             MNIST(20 epoch)              |                  fashion-mnist(20 epoch) |
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image |   ![LSGAN_MNIST_img](./result_images/LSGAN-MNIST.png)   | ![LSGAN_Fashion-MNIST_img](./result_images/LSGAN-fashion_mnist.png) |
|    loss D    | ![LSGAN_MNIST_loss-d](./result_images/LSGAN-MNIST-loss_D.png ) | ![LSGAN_Fashion-MNIST_loss-d](./result_images/LSGAN-fashion-mnist-loss_D.png) |
| loss D_real  | ![LSGAN_MNIST_loss-d-real](./result_images/LSGAN-MNIST-loss_D_real.png ) | ![LSGAN_Fashion-MNIST_loss-d-real](./result_images/LSGAN-fashion-mnist-loss_D_real.png) |
|  loss D_gen  | ![LSGAN_MNIST_loss-d-gen](./result_images/LSGAN-MNIST-loss_D_gen.png ) | ![LSGAN_Fashion-MNIST_loss-d-gen](./result_images/LSGAN-fashion-mnist-loss_D_gen.png) |
|    loss G    | ![LSGAN_MNIST_loss-g](./result_images/LSGAN-MNIST-loss_G.png ) | ![LSGAN_Fashion-MNIST_loss-g](./result_images/LSGAN-fashion-mnist-loss_G.png) |

|   dataset    |            CIFAR10(20 epoch)             |                            LLD(20 epoch) |
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image |  ![LSGAN_CIFAR10_img](./result_images/LSGAN-CIFAR10.png)  |       ![LSGAN_LLD_img](./result_images/LSGAN-LLD.png) |
|    loss D    | ![LSGAN_CIFAR10_loss-d](./result_images/LSGAN-CIFAR10-loss_D.png) | ![LSGAN_LLD_loss-d](./result_images/LSGAN-LLD-loss_D.png) |
| loss D_real  | ![LSGAN_CIFAR10_loss-d-real](./result_images/LSGAN-CIFAR10-loss_D_real.png) | ![LSGAN_LLD_loss-d-real](./result_images/LSGAN-LLD-loss_D_real.png) |
|  loss D_gen  | ![LSGAN_CIFAR10_loss-d-gen](./result_images/LSGAN-CIFAR10-loss_D_gen.png ) | ![LSGAN_LLD_loss-d-gen](./result_images/LSGAN-LLD-loss_D_gen.png) |
|    loss G    | ![LSGAN_CIFAR10_loss-g](./result_images/LSGAN-CIFAR10-loss_G.png ) | ![LSGAN_LLD_loss-g](./result_images/LSGAN-LLD-loss_G.png) |

### WGAN

|   dataset    |             MNIST(20 epoch)              |                  fashion-mnist(20 epoch) |
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image |   ![WGAN_MNIST_img](./result_images/WGAN-MNIST.png)    | ![LSGAN_Fashion-MNIST_img](./result_images/WGAN-fashion_mnist.png) |
|    loss D    | ![WGAN_MNIST_loss-d](./result_images/WGAN-MNIST-loss_D.png ) | ![LSGAN_Fashion-MNIST_loss-d](./result_images/WGAN-fashion-mnist-loss_D.png) |
| loss D_real  | ![WGAN_MNIST_loss-d-real](./result_images/WGAN-MNIST-loss_D_real.png ) | ![LSGAN_Fashion-MNIST_loss-d-real](./result_images/WGAN-fashion-mnist-loss_D_real.png) |
|  loss D_gen  | ![WGAN_MNIST_loss-d-gen](./result_images/WGAN-MNIST-loss_D_gen.png ) | ![LSGAN_Fashion-MNIST_loss-d-gen](./result_images/WGAN-fashion-mnist-loss_D_gen.png) |
|    loss G    | ![WGAN_MNIST_loss-g](./result_images/WGAN-MNIST-loss_G.png ) | ![LSGAN_Fashion-MNIST_loss-g](./result_images/WGAN-fashion-mnist-loss_G.png) |

| dataset      | CIFAR10(20 epoch) | LLD(20 epoch)|
| :----------: | :--------------------------------------: | :--------------------------------------: |
| result image | ![WGAN_CIFAR10_img](./result_images/WGAN-CIFAR10.png)       |![WGAN_LLD_img](./result_images/WGAN-LLD.png)    |
| loss D       | ![WGAN_CIFAR10_loss-d](./result_images/WGAN-CIFAR10-loss_D.png) | ![WGAN_LLD_loss-d](./result_images/WGAN-LLD-loss_D.png) |
| loss D_real  | ![WGAN_CIFAR10_loss-d-real](./result_images/WGAN-CIFAR10-loss_D_real.png) | ![WGAN_LLD_loss-d-real](./result_images/WGAN-LLD-loss_D_real.png) |
| loss D_gen   | ![WGAN_CIFAR10_loss-d-gen](./result_images/WGAN-CIFAR10-loss_D_gen.png ) | ![WGAN_LLD_loss-d-gen](./result_images/WGAN-LLD-loss_D_gen.png) |
| loss G       | ![WGAN_CIFAR10_loss-g](./result_images/WGAN-CIFAR10-loss_G.png ) | ![WGAN_LLD_loss-g](./result_images/WGAN-LLD-loss_G.png) |

| ![after 5 epoch](./result_images/WGAN-LLD_35000iter.png ) |
| :----------: |
| WGAN result - after training LLD dataset for 5 epoch|

## Observation

`MNIST` dataset is easy for `GAN`, `LSGAN`, `WGAN`.

Training with `Fashion-MNIST` dataset, **`GAN`** is slightly better others.

Training with `CIFAR 10` dataset, **`WGAN`** are better than others. but clearly does not generate looks pretty image like original image.

Training with `LLD` dataset, three GANs generate **similar** image.
Compare with original image, three GANs generate ugly image.
Compare other GANs, WGAN generate best image in less epoch, but after 5 epoch generate worse than others.
Above loss of generator and discriminator, generator overpowered discriminator.

GANs trained outline of all original dataset(fashion-mnist, CIFAR10, LLD), but did not train detail.

## Getting Started

1. `python ./setup.py install` and follow console (If you already install dependencies, follow step 2.)
2. Run bench code `python ./main.py`
3. If you want to run another model, modify `workbench/bench_code.py`.

## Folder structure

```terminal
├─data          # default dataset
├─data_handler
├─dict_keys
├─instance      # default model instance
├─model
├─unit_test
├─util
├─visualizer
└─workbench     # bench code, DatasetHelper
```

## Dependency

- Python 3.5+
  - matplotlib (2.1.2)
  - numpy (1.14.0)
  - pandas (0.22.0)
  - Pillow (5.0.0)
  - scikit-image (0.13.1)
  - scikit-learn (0.19.1)
  - scipy (1.0.0)
  - tensorflow (1.4.1)
  - tensorflow-gpu (1.4.1)
  - tensorflow-tensorboard (0.4.0)
  - opencv-python (3.4.0.12)
  - requests (2.18.4)

## Authors

- [demetoir](https://github.com/demetoir)
  - e-mail: wnsqlehlswk@naver.com
- [WKBae](https://github.com/WKBae)
  - e-mail: williambae1@gmail.com
- [StarG](https://github.com/psk7142)
  - e-mail: psk7142@naver.com

## Reference

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
