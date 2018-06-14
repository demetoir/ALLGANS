from sklearn_like_toolkit.Base import BaseClass


class BaseTFWrapperSklearn(BaseClass):
    def compile_graph(self, input_shapes):
        pass

    def get_tf_values(self, fetches, feed_dict):
        pass

    def save(self):
        pass

    def load(self):
        pass


class tf_GAN(BaseTFWrapperSklearn):
    def fit(self, Xs):
        pass

    def generate(self, zs):
        pass


class tf_C_GAN(BaseTFWrapperSklearn):
    def fit(self, Xs, Ys):
        pass

    def generate(self, zs, Ys):
        pass


class tf_info_GAN(BaseTFWrapperSklearn):
    def fit(self, Xs, Ys):
        pass

    def generate(self, zs, Ys):
        pass


class tf_AE(BaseTFWrapperSklearn):
    def fit(self, Xs):
        pass

    def code(self, Xs):
        pass

    def recon(self, zs):
        pass


class tf_VAE(BaseTFWrapperSklearn):
    def fit(self, Xs):
        pass

    def encode(self, Xs):
        pass

    def decode(self, zs):
        pass


class tf_AAE(BaseTFWrapperSklearn):
    def fit(self, Xs):
        pass

    def code(self, Xs):
        pass

    def recon(self, Xs, zs):
        pass


class tf_AAEClassifier(BaseTFWrapperSklearn):
    def fit(self, Xs):
        pass

    def code(self, Xs):
        pass

    def recon(self, Xs):
        pass

    def predict(self, Xs):
        pass

    def score(self, Xs, Ys):
        pass

    def proba(self, Xs):
        pass


class tf_MLPClassifier(BaseTFWrapperSklearn):
    def fit(self, Xs, Ys):
        pass

    def predict(self, Xs):
        pass

    def score(self, Xs, Ys):
        pass

    def proba(self, Xs):
        pass
