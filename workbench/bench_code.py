# -*- coding:utf-8 -*-


########################################################################################################################
# print(built-in function) is not good for logging

from util.Logger import StdoutOnlyLogger, pprint_logger

bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)


########################################################################################################################


def main():
    pass
