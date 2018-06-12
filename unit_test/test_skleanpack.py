from pprint import pprint
import numpy as np
import os


def finger_print(size, head=''):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


class test_clf_pack:

    def setup(self):
        print('reset current dir')

        print('cur dir')
        print(os.getcwd())
        head, tail = os.path.split(os.getcwd())
        os.chdir(head)
        print(os.getcwd())

        from data_handler.DatasetLoader import DatasetLoader
        from sklearn_like_toolkit.ClassifierPack import ClassifierPack

        self.cls = ClassifierPack

        dataset = DatasetLoader().load_dataset("titanic")
        train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])
        self.dataset = dataset
        self.train_Xs = train_Xs
        self.train_Ys = train_Ys
        self.valid_Xs = valid_Xs
        self.valid_Ys = valid_Ys

    def test_0_save_params(self):
        ClassifierPack = self.cls
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

    def test_1_load_params(self):
        ClassifierPack = self.cls
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

    def test_hard_voting(self):
        ClassifierPack = self.cls
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf = ClassifierPack()
        clf = clf.make_FoldingHardVote()
        clf.fit(train_Xs, train_Ys)

        predict = clf.predict(valid_Xs[:4])
        print(f'predict {predict}')

        proba = clf.predict_proba(valid_Xs[:4])
        print(f'proba {proba}')

        predict_bincount = clf.predict_bincount(valid_Xs[:4])
        print(f'predict_bincount {predict_bincount}')

        score = clf.score(valid_Xs, valid_Ys)
        print(f'score {score}')

    def test_stacking(self):
        ClassifierPack = self.cls
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf = ClassifierPack()
        metaclf = clf.pack['XGBoost']
        clf = clf.make_stackingClf(metaclf)
        clf.fit(train_Xs, train_Ys)

        predict = clf.predict(valid_Xs[:4])
        print(f'predict {predict}')

        proba = clf.predict_proba(valid_Xs[:4])
        print(f'proba {proba}')

        score = clf.score(valid_Xs, valid_Ys)
        print(f'score {score}')
        pass

    def test_stackingCV(self):
        ClassifierPack = self.cls
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf = ClassifierPack()
        metaclf = clf.pack['XGBoost']
        clf = clf.make_stackingCVClf(metaclf)
        clf.fit(train_Xs, train_Ys)

        predict = clf.predict(valid_Xs[:4])
        print(f'predict {predict}')

        proba = clf.predict_proba(valid_Xs[:4])
        print(f'proba {proba}')

        score = clf.score(valid_Xs, valid_Ys)
        print(f'score {score}')

    def test_clf_pack_param_search(self):
        ClassifierPack = self.cls
        dataset = self.dataset
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf = ClassifierPack()
        clf.param_search(train_Xs, train_Xs)

        dataset.shuffle()
        train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

        pprint('train score', clf.score(train_Xs, train_Ys))
        pprint('test score', clf.score(valid_Xs, valid_Ys))
        pprint('predict', clf.predict(valid_Xs[:2]))
        pprint('predict_proba', clf.predict_proba(valid_Xs[:2]))

    def test_clfpack(self):
        ClassifierPack = self.cls
        dataset = self.dataset
        train_Xs = self.train_Xs
        train_Ys = self.train_Ys
        valid_Xs = self.valid_Xs
        valid_Ys = self.valid_Ys

        clf = ClassifierPack()
        clf.fit(train_Xs, train_Ys)
        predict = clf.predict(valid_Xs[:2])
        pprint('predict', predict)
        proba = clf.predict_proba(valid_Xs[:2])
        pprint('predict_proba', proba)

        score = clf.score(valid_Xs, valid_Ys)
        pprint('test score', score)

        score_pack = clf.score_pack(valid_Xs, valid_Ys)
        pprint('score pack', score_pack)

        # initialize data
