from pprint import pprint
import numpy as np
from data_handler.DatasetLoader import DatasetLoader
from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from sklearn_like_toolkit.sklearn_toolkit import VotingClassifier


def finger_print(size, head=''):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


class test_clf_pack:
    def __init__(self):
        dataset = DatasetLoader().load_dataset("titanic")
        train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

        self.dataset = dataset
        self.train_Xs = train_Xs
        self.train_Ys = train_Ys
        self.valid_Xs = valid_Xs
        self.valid_Ys = valid_Ys

    def test_save_params(self):
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf_pack = ClassifierPack(['skMLP'])
        clf_pack.param_search(train_Xs, train_Ys)
        train_score = clf_pack.score(train_Xs, train_Ys)
        valid_score = clf_pack.score(valid_Xs, valid_Ys)
        pprint(train_score)
        pprint(valid_score)

    def test_load_params(self):
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf_pack = ClassifierPack(['skMLP'])
        clf_pack.param_search(train_Xs, train_Ys)
        train_score = clf_pack.score(train_Xs, train_Ys)
        valid_score = clf_pack.score(valid_Xs, valid_Ys)

        pprint(train_score)
        pprint(valid_score)

        path = clf_pack.save_params()
        clf_pack.load_params(path)
        clf_pack.fit(train_Xs, train_Ys)
        train_score = clf_pack.score(train_Xs, train_Ys)
        valid_score = clf_pack.score(valid_Xs, valid_Ys)
        pprint(train_score)
        pprint(valid_score)

    def test_voting(self):
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        key = 'LightGBM'
        clf_pack = ClassifierPack()
        pack = clf_pack.pack

        # pprint(pack)
        # pprint(list(zip(*pack)))

        clf_list = []
        for i in range(3):
            clf_list += [(k + finger_print(8, head='_'), v) for k, v in pack.items()]

        pprint(clf_list)

        # clf_list = [(k, v) for k, v in pack.items()] * 10
        clf = VotingClassifier(clf_list, voting='soft')
        clf.fit(train_Xs, train_Ys)
        score = clf.score(valid_Xs, valid_Ys)
        pprint(score)

    def run(self):
        self.test_load_params()
        self.test_save_params()
        self.test_voting()


if __name__ == '__main__':
    test_class = test_clf_pack()
    test_class.run()
