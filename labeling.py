# -*- coding: utf-8 -*-
from sklearn.cluster import k_means
import random
import PIL.Image

img_sample = None


class Labeling:
    pass


def grid_image(images, col=20):
    width = 32 * col
    height = 32 * (len(images) // col + 1)
    grid = PIL.Image.new('RGB', (width, height), 'black')

    for idx, img in enumerate(images):
        x = idx % col
        y = idx // col
        offset = (32 * x, 32 * y)
        grid.paste(img, offset)

    return grid
    pass


def rgb_list(img):
    w, h = img.size
    rgb = []
    for i in range(w):
        for j in range(h):
            rgb += [img.getpixel((i, j))]

    return rgb


def img_rgb_clustering(img, max_n_clusters=10):
    def get_square_error(dist, centroid, label):
        raise NotImplementedError

    img = img_sample[random.randint(1, len(img_sample))]
    rgb_dist = rgb_list(img)

    for n_clusters in range(2, max_n_clusters + 1):
        centroid, label, _ = k_means(rgb_dist, n_clusters=n_clusters)
        get_square_error(rgb_dist, centroid, label)

    def elbow_method(centroid, label, data):
        raise NotImplementedError

    raise NotImplementedError


if __name__ == '__main__':
    # icons = LLD().load()
    # grid = grid_image(icons[1000:1777])
    # grid.show()

    # img_sample = LLD.load_sample()
    print(img_sample)
    img = img_sample[3]
    img.show()

    #
    # for i in range(len(label)):
    #     print(rgb_dist[i], label[i])
    # pass
    # 색상 별로 구분하기..
