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
import tensorflow as tf

# print(built-in function) is not good for logging
from util.misc_util import dump_json, load_json


def pprint_logger(log_func):
    def wrapper(*args, **kwargs):
        import pprint
        return log_func(pprint.pformat(args, **kwargs))

    return wrapper


bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)


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
        ('log_confusion_matrix', 400),
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


def model_test():
    tf.logging.set_verbosity(0)
    dataset = DatasetLoader().load_dataset("titanic")
    input_shapes = dataset.train_set.input_shapes
    from model.Classifier.MLPClassifier import MLPClassifier
    # model = ModelClassLoader.load_model_class('MLPClassifier')

    Xs, Ys = dataset.train_set.full_batch(
        batch_keys=["Xs", "Ys"],
    )

    model = MLPClassifier(input_shapes)
    model.build()
    model.train(Xs, Ys, epoch=10)
    # model.train_dataset(dataset.train_set, epoch=5)
    # model.train(Xs, Ys, epoch=2)
    path = model.save()

    Xs, Ys = dataset.train_set.next_batch(
        10,
        batch_keys=["Xs", "Ys"],
    )

    predict = model.predict(Xs)
    print(predict)

    loss = model.metric(Xs, Ys)
    print(loss)

    proba = model.proba(Xs)
    print(proba)

    score = model.score(Xs, Ys)
    print(score)

    # del model
    # model = MLPClassifier(input_shapes)
    # model.load(path)
    # model.train_dataset(dataset.train_set, epoch=1)


def tf_data_pipeline():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Wrapping all together -> Switch between train and test set
        EPOCHS = 10
        BATCH_SIZE = 16

        # create a placeholder to dynamically switch between batch sizes
        batch_size = tf.placeholder(tf.int64)
        x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])

        # using two numpy arrays
        train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
        Xs, Ys = train_data
        test_data = (np.random.sample((20, 2)), np.random.sample((20, 1)))
        n_batches = len(train_data[0]) // BATCH_SIZE

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
        iter = dataset.make_initializable_iterator()
        features, labels = iter.get_next()
        # initialise iterator with train data
        sess.run(iter.initializer, feed_dict={
            x: Xs,
            y: Ys,
            batch_size: BATCH_SIZE
        })

        # make a simple model
        net = tf.layers.dense(features, 8, activation=tf.tanh)  # pass the first value from iter.get_next() as input
        net = tf.layers.dense(net, 8, activation=tf.tanh)
        prediction = tf.layers.dense(net, 1, activation=tf.tanh)
        loss = tf.losses.mean_squared_error(prediction, labels)  # pass the second value from iter.get_net() as label
        train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.01).minimize(loss)

        print('Training...')
        for i in range(EPOCHS):
            tot_loss = 0
            for _ in range(n_batches):
                _, loss_value = sess.run([train_op, loss])
                tot_loss += loss_value
            print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))

        # initialise iterator with test data
        sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        print('Test Loss: {:4f}'.format(sess.run(loss)))


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


def main():
    size = 100
    urls = []
    base_format = """http://gall.dcinside.com/board/lists/?id=disease&page={}"""
    for i in range(1, size + 1):
        url = base_format.format(i)
        urls += [url]
        print(url)

    from workbench.crawler import Crawler
    crawler = Crawler()
    ret = crawler.run(urls)
    path = os.path.join('.', 'crawl_result', 'result.txt')
    crawler.save(ret, path)

    # dataset = DatasetLoader().load_dataset("MNIST")
    # dataset = DatasetLoader().load_dataset("CIFAR10")
    # dataset = DatasetLoader().load_dataset("CIFAR100")
    # dataset = DatasetLoader().load_dataset("Fashion_MNIST")
    # model_test()

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

    # tf_model_train(**MLPClassifier)
