# -*- coding:utf-8 -*-


########################################################################################################################
# print(built-in function) is not good for logging
from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from sklearn_like_toolkit.FoldingHardvote import FoldingHardVote

from data_handler.DatasetLoader import DatasetLoader
from util.Logger import StdoutOnlyLogger, pprint_logger
import numpy as np
import pandas as pd
from tqdm import trange

from util.misc_util import path_join

bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)


#######################################################################################################################


def baseline_clf():
    # predict_Ys = clf_pack.predict(test_Xs)[key]
    # dataset.to_kaggle_submit_csv(path_join('.', 'submit.csv'), predict)

    dataset = DatasetLoader().load_dataset("titanic")
    test_set = dataset.set['test']
    test_Xs = test_set.full_batch(['Xs'])

    n_iter = 1
    key = 'LightGBM'

    test_predicts = []
    train_predicts = []
    for _ in trange(n_iter):
        dataset.shuffle()
        train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

        clf_pack = ClassifierPack()
        # clf_pack.load_params(path)
        clf_pack.fit(train_Xs, train_Ys)
        train_score = clf_pack.score(train_Xs, train_Ys)
        valid_score = clf_pack.score(valid_Xs, valid_Ys)
        pprint(train_score, valid_score)

        dataset.merge('train', 'valid', 'train')
        dataset.sort()
        train_set = dataset.set['train']
        train_Xs = train_set.full_batch(['Xs'])

        test_predict = clf_pack.predict(test_Xs)[key]
        test_predicts += [test_predict]

        train_predict = clf_pack.predict(train_Xs)[key]
        train_predicts += [train_predict]

        prob = clf_pack.predict_proba(train_Xs)
        pprint(prob)
        pprint(clf_pack.predict(train_Xs))

    test_sum = sum(test_predicts)
    train_sum = sum(train_predicts)

    df = pd.DataFrame()
    df['sum'] = test_sum
    df.to_csv(path_join('.', 'predict_sum.csv'), index=False)

    df = pd.DataFrame()
    df['sum'] = train_sum
    train_Ys = train_set.full_batch(['Survived'])
    pprint(train_Ys)
    df['Y'] = train_Ys
    df.to_csv(path_join('.', 'train_sum.csv'), index=False)

    def count_mid(Xs, threashold=3):
        pprint(Xs)
        max_val = max(Xs)
        print(max_val)
        count = 0
        for x in Xs:
            if threashold < x < max_val - threashold:
                count += 1
        return count

    hold = 10
    # print("count mid train {}".format(count_mid(train_sum, hold)))
    # print("count mid test {}".format(count_mid(test_sum, hold)))

    filted_train = filter(lambda a: True if hold < a < 100 - hold else False, train_sum)
    filted_train = list(sorted(filted_train))
    filted_test = filter(lambda a: True if hold < a < 100 - hold else False, test_sum)
    filted_test = list(sorted(filted_test))
    # pprint(filted_train)
    # pprint(filted_test)
    #
    # plt.hist(filted_test)
    # plt.show()
    # plt.hist(filted_train)
    # plt.show()


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def finger_print(size, head='_'):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


def test_clf_pack_param_search():
    dataset = DatasetLoader().load_dataset("titanic")

    clf = ClassifierPack()
    train_set = dataset.set['train']
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    clf.param_search(train_Xs, train_Xs)

    dataset.shuffle()
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    pprint('train score', clf.score(train_Xs, train_Ys))
    pprint('test score', clf.score(valid_Xs, valid_Ys))
    pprint('predict', clf.predict(valid_Xs[:2]))
    pprint('predict_proba', clf.predict_proba(valid_Xs[:2]))


def test_foldVote():
    dataset = DatasetLoader().load_dataset("titanic")
    dataset.shuffle()
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    lightgbm = clf.pack['LightGBM']

    base_clf = lightgbm
    base_clf.fit(train_Xs, train_Ys)
    score = base_clf.score(valid_Xs, valid_Ys)
    print(f'base score {score}')

    clf = FoldingHardVote(lightgbm, 1000)
    clf.fit(train_Xs, train_Ys)

    predict = clf.predict(valid_Xs)
    # print(f'predict {predict}')

    proba = clf.predict_proba(valid_Xs)
    # print(f'proba {proba}')

    predict_bincount = clf.predict_bincount(valid_Xs)
    # print(f'predict_bincount {predict_bincount}')

    score = clf.score(valid_Xs, valid_Ys)
    print(f'score {score}')


def main():
    ret = np.bincount([1, 2, 3, 5], minlength=7)
    print(ret)
    # test_pool()
    # test_clfpack()
    # test_clf_pack_param_search()
    test_foldVote()
    pass
