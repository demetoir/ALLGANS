# -*- coding:utf-8 -*-


########################################################################################################################
# print(built-in function) is not good for logging
from util.Logger import StdoutOnlyLogger, pprint_logger
from unit_test.test_BaseModel import *

bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)


########################################################################################################################


def main():
    # test_MLPClassifier()
    # test_AE()
    # test_VAE()
    # test_CVAE()
    # test_DAE()
    # test_DVAE()

    test_AAE()
    pass

