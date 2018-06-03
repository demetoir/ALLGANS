from data_handler.DatasetLoader import DatasetLoader
import numpy as np

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

def model_test():
    dataset = DatasetLoader().load_dataset("titanic")
    input_shapes = dataset.train_set.input_shapes
    from model.sklearn_like_model.MLPClassifier import MLPClassifier
    # model = ModelClassLoader.load_model_class('MLPClassifier')

    Xs, Ys = dataset.train_set.full_batch(
        batch_keys=["Xs", "Ys"],
    )

    model = MLPClassifier(input_shapes)
    model.build()
    model.train(Xs, Ys, epoch=10)
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

    model = MLPClassifier(input_shapes)
    model.load(path)


def test_sklike_AE_test():
    from data_handler.DatasetLoader import DatasetLoader
    from model.sklearn_like_model.AutoEncoder import AutoEncoder

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
