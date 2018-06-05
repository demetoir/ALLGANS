from model.sklearn_like_model.AE.AAE import AAE
from model.sklearn_like_model.AE.CVAE import CVAE
from model.sklearn_like_model.AE.DVAE import DVAE
from model.sklearn_like_model.AE.VAE import VAE
from model.sklearn_like_model.AE.DAE import DAE
from model.sklearn_like_model.MLPClassifier import MLPClassifier
from model.sklearn_like_model.AE.AutoEncoder import AutoEncoder
from data_handler.DatasetLoader import DatasetLoader
import numpy as np


class test_MLPClassifier:
    def __init__(self):
        self.test()

    def test(self):
        dataset = DatasetLoader().load_dataset("titanic")
        input_shapes = dataset.train_set.input_shapes

        Xs, Ys = dataset.train_set.full_batch(
            batch_keys=["Xs", "Ys"],
        )

        model = MLPClassifier(input_shapes)
        model.build()
        model.train(Xs, Ys, epoch=1)

        Xs, Ys = dataset.train_set.next_batch(
            5,
            batch_keys=["Xs", "Ys"],
        )

        predict = model.predict(Xs)
        print("predict {}".format(predict))

        loss = model.metric(Xs, Ys)
        print("loss {}".format(loss))

        proba = model.proba(Xs)
        print('prob {}'.format(proba))

        score = model.score(Xs, Ys)
        print('score {}'.format(score))

        path = model.save()
        model = MLPClassifier()
        model.load(path)

        predict = model.predict(Xs)
        print("predict {}".format(predict))

        loss = model.metric(Xs, Ys)
        print("loss {}".format(loss))

        proba = model.proba(Xs)
        print('prob {}'.format(proba))

        score = model.score(Xs, Ys)
        print('score {}'.format(score))


class test_AE:
    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set

        model = AutoEncoder(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = AutoEncoder()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set

        model = AutoEncoder(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = AutoEncoder()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))


class test_VAE:
    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set

        model = VAE(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = VAE()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set

        model = VAE(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = VAE()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))


class test_DAE:
    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        class_ = DAE
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        class_ = DAE

        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))


class test_CVAE:
    class_ = CVAE

    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()
        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X, sample_Y)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X, sample_Y)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()
        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X, sample_Y)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X, sample_Y)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))


class test_DVAE:
    class_ = DVAE

    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

    def test_titanic(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set

        model = class_(dataset.input_shapes)
        model.build()

        Xs = dataset.full_batch(['Xs'])
        model.train(Xs, epoch=1)

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        sample_X = Xs[:2]
        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(Xs)
        loss = np.mean(loss)
        print("loss {:.4}".format(loss))


class test_AAE:
    class_ = AAE

    def __init__(self):
        self.test_mnist()
        self.test_titanic()

    def test_mnist(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("MNIST")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

    def test_titanic(self):
        class_ = self.class_
        dataset = DatasetLoader().load_dataset("titanic")
        dataset = dataset.train_set
        Xs, Ys = dataset.full_batch(['Xs', 'Ys'])
        sample_X = Xs[:2]
        sample_Y = Ys[:2]

        model = class_(dataset.input_shapes)
        model.build()

        model.train(Xs, Ys, epoch=1)

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {:}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))

        path = model.save()

        model = class_()
        model.load(path)
        print('model reloaded')

        code = model.code(sample_X)
        print("code {code}".format(code=code))

        recon = model.recon(sample_X, sample_Y)
        print("recon {recon}".format(recon=recon))

        loss = model.metric(sample_X, sample_Y)
        print("loss {:}".format(loss))

        # generate(self, zs, Ys)

        proba = model.proba(sample_X)
        print("proba {}".format(proba))

        predict = model.predict(sample_X)
        print("predict {}".format(predict))

        score = model.score(sample_X, sample_Y)
        print("score {}".format(score))


if __name__ == '__main__':
    pass
