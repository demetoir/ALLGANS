# -*- coding:utf-8 -*-
from util.Logger import StdoutOnlyLogger

# print(built-in function) is not good for logging


########################################################################################################################
from workbench.ppomppu import pomppu_page_crawl


def pprint_logger(log_func):
    def wrapper(*args, **kwargs):
        import pprint
        return log_func(pprint.pformat(args, **kwargs))

    return wrapper


bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)


########################################################################################################################


def main():
    pomppu_page_crawl()
    pass
