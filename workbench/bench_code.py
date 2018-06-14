# -*- coding:utf-8 -*-


########################################################################################################################
# print(built-in function) is not good for logging
from sklearn_like_toolkit.ClassifierPack import ClassifierPack

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


def exp_stacking_metaclf():
    dataset = DatasetLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.make_stackingClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score(valid_Xs, valid_Ys)
        pprint(f'score {score}')


def exp_voting():
    dataset = DatasetLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    clf.fit(train_Xs, train_Ys)
    score_pack = clf.score_pack(valid_Xs, valid_Ys)
    pprint(score_pack)

    voting = clf.make_FoldingHardVote()
    voting.fit(train_Xs, train_Ys)
    score = voting.score(valid_Xs, valid_Ys)
    pprint(score)
    bincount = voting.predict_bincount(valid_Xs[:5])
    pprint(bincount)

    # metaclf = clf.pack['LightGBM']
    # stacking = clf.make_stackingClf(metaclf)

    """
    default param clf pack
    default param clf pack to hard voting
    default param clf pack to stacking
    dcfault praam clf pack to stacking cv

    default param clf pack * 10 to hard voting    
    default param clf pack * 10 to stacking    
    default param clf pack * 10 to stacking cv
    
    default param clf pack * 100 to hard voting    
    default param clf pack * 100 to stacking    
    default param clf pack * 100 to stacking cv
    
    optimize param clf pack top1 
    optimize param clf pack top1 to hard voting
    optimize param clf pack top1 to stacking
    optimize param clf pack top1 to stacking cv
    
    optimize param clf pack top1 *10 to hard voting
    optimize param clf pack top1 *10 to stacking
    optimize param clf pack top1 *10 to stacking cv
    
    optimize param clf pack top5 *100 to hard voting
    optimize param clf pack top5 *100 to stacking
    optimize param clf pack top5 *100 to stacking cv
    
    out 
    score,
    score_pack
    predict 
    
    csv to save 
    
    
    
    
    
    
    
    
    
        
    
    """

    # todo
    # voting and check recall p

    pass


def exp_titanic_statistic():
    # todo
    # variant clf
    # check which clf predict each data with id

    # result save to csv

    # fit clfs

    # predict clfs

    # save predic

    pass


def main():
    print(exp_titanic_statistic.__name__)
    # exp_stacking_metaclf()
    # exp_voting()
    pass
