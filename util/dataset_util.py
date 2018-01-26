from data_handler.LLD import LLD
from PIL import Image
from glob import glob
import numpy as np
import pickle

from env_settting import *


DECOMPOSED_PATH = os.path.join(ROOT_PATH, 'dataset', 'LLD_decomposed')


def LLD_mkdir():
    if not os.path.exists(DECOMPOSED_PATH):
        os.mkdir(DECOMPOSED_PATH)

    packed_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_packed')
    if not os.path.exists(packed_path):
        os.mkdir(packed_path)
    packed_path = os.path.join(ROOT_PATH, 'dataset', 'LLD_packed', 'packed.pkl')


def decompose_to_imgs(DECOMPOSED_PATH, batch):
    img_num = 0
    for img_np in batch:
        img_num += 1
        img = Image.fromarray(img_np)
        img_name = str(img_num).zfill(3) + '.jpg'
        with open(os.path.join(DECOMPOSED_PATH, img_name), mode='w') as f:
            img.save(f)


def pack_img(img_path, packed_path):
    print('collecting img')
    files = glob(os.path.join(img_path, '*', '*.jpg'))
    files.sort()
    print('collect', len(files))

    packed = []
    for file in files:
        with open(file, mode='rb') as fp:
            img_np = np.asarray(Image.open(fp))
            packed += [img_np]
    packed = np.array(packed)
    print('pack img')

    with open(packed_path, mode='wb') as fp:
        pickle.dump(packed, fp)
    print('dump pkl')


def show_pkl_img(path):
    with open(path, 'rb') as fp:
        imgs_np = pickle.load(fp)

    img_np = imgs_np[3]

    img = Image.fromarray(img_np)
    img.show()


def bucket_decompose_img(bucket_number, bucket_size, dataset):
    for i in range(bucket_number):
        path = os.path.join(DECOMPOSED_PATH, str(i).zfill(4))
        if not os.path.exists(path):
            os.mkdir(path)
        decompose_to_imgs(path, lld_data.next_batch(bucket_size))
        print('decompose %d / %d' % (i + 1, bucket_number))
        print('decompose at %s' % path)


if __name__ == '__main__':
    LLD_PATH = os.path.join(ROOT_PATH, 'dataset', 'LLD')
    lld_data = LLD()
    lld_data.load(LLD_PATH)

    bucket_decompose_img(1, 50, lld_data)

    # pack_img(DECOMPOSED_PATH, packed_path)
