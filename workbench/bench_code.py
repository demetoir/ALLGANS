# -*- coding:utf-8 -*-
import codecs
import webbrowser
from bs4 import BeautifulSoup

from InstanceManger import InstanceManager
from DatasetLoader import DatasetLoader
from VisualizerClassLoader import VisualizerClassLoader
from ModelClassLoader import ModelClassLoader
from data_handler.titanic import *
from sklearn_like_toolkit.sklearn_toolkit import ClassifierPack
from util.Logger import StdoutOnlyLogger

# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR10")
# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR100")
# dataset, input_shapes = DatasetLoader().load_dataset("LLD")
# dataset, input_shapes = DatasetLoader().load_dataset("MNIST")
# dataset, input_shapes = DatasetLoader().load_dataset("Fashion_MNIST")
# model = ModelClassLoader.load_model_class("GAN")


# print(built-in function) is not good for logging
bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()


def fit_and_test(model, train_Xs, train_Ys, test_Xs, test_Ys):
    instance = model()
    print(instance)

    instance.fit(train_Xs, train_Ys, Ys_type="onehot")

    acc = instance.score(train_Xs, train_Ys, Ys_type="onehot")
    print("train acc:", acc)

    acc = instance.score(test_Xs, test_Ys, Ys_type="onehot")
    print("valid acc:", acc)

    print("probs")
    probs = instance.proba(test_Xs[:10], transpose_shape=False)
    print(probs)

    print("predict")
    print(instance.predict(test_Xs[:10]))
    print()


def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text
    return _html


def deco_file_lines_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
    def wrapper():
        with codecs.open(in_file, 'r', encoding=encoding) as f:
            lines = [line for line in f.readlines()]

        print("read '{}', {} lines".format(in_file, len(lines)))

        lines = func(lines)

        if lines is not None:
            with codecs.open(out_file, 'w', encoding=encoding) as f:
                f.writelines(lines)
            print("write '{}', {} lines".format(out_file, len(lines)))

    return wrapper


def deco_file_str_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
    def wrapper():
        with codecs.open(in_file, 'r', encoding=encoding) as f:
            line = "".join([line for line in f.readlines()])

        print("read '{}', {} lenght".format(in_file, len(line)))

        line = func(line)

        if line is not None:
            with codecs.open(out_file, 'w', encoding=encoding) as f:
                f.writelines(str(line))
            print("write '{}', {} lenght".format(out_file, len(line)))

    return wrapper


def filter_str(s, filter_):
    for del_char in filter_:
        s = s.replace(del_char, '')
    return s


@deco_file_lines_job
def filter_lines(lines):
    out = []
    for line in lines:
        print("%s [ %s ]" % (len(line), line))
        line = filter_str(line, """+-():?"':!~,‘’""")
        # if len(line) == 2 or len(line) > 30 or '[' in line or '.' in line:
        #     continue
        out += [line]

        print("%s [ %s ]" % (len(line), line))

    out = set(out)
    out = list(out)
    out = sorted(out)
    return out


@deco_file_lines_job
def open_chrome(lines):
    batch_size = 5
    for idx in range(0, len(lines), batch_size):
        url_head = "https://www.google.co.kr/search?q=%s"
        for token in lines[idx:idx + batch_size]:
            url = url_head % token
            print(idx, url)
            webbrowser.open(url)

        input("enter to open next batch")


@deco_file_lines_job
def get_google_cite_url(lines):
    out = []
    for line in lines:
        # url = "https://www.google.co.kr/search?q=%s" + line
        url = line

        print(line)

        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')

        cites = soup.find_all("cite")
        print(cites[0].text)
        out += [cites[0].text]


@deco_file_str_job
def filter_book_mark(html):
    # print(html)
    soup = BeautifulSoup(html, 'html.parser')
    print(len(soup.find_all('a')))
    out = []
    for i in soup.find_all('a'):
        # if i is None: continue
        # if hasattr(i, 'icon'):
        #     i['icon'] = None
        #     i['ICON'] = None
        # print()
        out += [i['href']]
        # print(a)
    # print(soup)

    out = sorted(out)
    return "\n".join(out)
    # return html


GAN = {
    "model": "GAN",
    "dataset": "MNIST",
    "visuliziers": [
        ('image_tile', 100),
        ('log_GAN_loss', 20),
    ],
    "epoch": 40

}

C_GAN = {
    "model": "C_GAN",
    "dataset": "MNIST",
    "visuliziers": [
        ('image_C_GAN', 100),
        ('log_C_GAN_loss', 20),
    ],
    "epoch": 40

}

info_GAN = {
    "model": "InfoGAN",
    "dataset": "MNIST",
    "visuliziers": [
        ('image_tile', 100),
        ('log_GAN_loss', 20),
    ],
    "epoch": 40

}

AE = {
    "model": "AE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AE', 100),
        ('image_AE', 100),
    ],
    "epoch": 40

}

VAE = {
    "model": "VAE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AE', 100),
        ('image_AE', 100),
    ],
    "epoch": 40

}

AAE = {
    "model": "AAE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AAE', 100),
        ('image_AE', 100),
        ('image_AAE_Ys', 100),
    ],
    "epoch": 40

}

DAE = {
    "model": "DAE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AAE', 100),
        ('image_DAE', 100),
    ],
    "epoch": 40

}

DVAE = {
    "model": "DAE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AAE', 100),
        ('image_DAE', 100),
    ],
    "epoch": 40

}

CVAE = {
    "model": "CVAE",
    "dataset": "MNIST",
    "visuliziers": [
        ('log_AE', 100),
        ('image_AE', 100),
        ('image_CVAE_Ys', 100),
    ],
    "epoch": 40

}

MLPClassifier = {
    "model": "MLPClassifier",
    "dataset": "titanic",
    "visuliziers": [
        ('log_titanic_loss', 25),
        ('log_confusion_matrix', 100),
    ],
    "epoch": 400

}


def tf_model_train(model=None, dataset=None, visuliziers=None, epoch=None):
    dataset = DatasetLoader().load_dataset(dataset)
    input_shapes = dataset.train_set.input_shapes
    model = ModelClassLoader.load_model_class(model)

    manager = InstanceManager()
    metadata_path = manager.build_instance(model, input_shapes)
    manager.load_instance(metadata_path)

    for v_fun, i in visuliziers:
        manager.load_visualizer(VisualizerClassLoader.load(v_fun), i)

    manager.train_instance(
        epoch=epoch,
        dataset=dataset,
        check_point_interval=5000,
        with_tensorboard=True
    )

    del manager


def main():
    # open_chrome()
    # filter_book_mark()
    #
    # dataset = DatasetLoader().load_dataset("titanic")
    # clfpack = ClassifierPack()
    # train_xs, train_labels = dataset.train_set.next_batch(
    #     dataset.train_set.data_size,
    #     batch_keys=[BK_X, BK_LABEL],
    # )
    # test_xs, test_labels = dataset.validation_set.next_batch(
    #     dataset.validation_set.data_size,
    #     batch_keys=[BK_X, BK_LABEL],
    # )
    # clfpack.param_search(train_xs, train_labels, test_xs, test_labels)

    # tf_model_train_GAN()
    # tf_model_train_C_GAN()
    # tf_model_train_infoGAN()
    #
    # tf_model_train_AE()
    # tf_model_train_VAE()
    # tf_model_train_AAE()
    # tf_model_train_DAE()
    # tf_model_train_DVAE()
    # tf_model_train_CVAE()
    #
    tf_model_train(**MLPClassifier)
    # tf_model_train_MLPClassifier()
