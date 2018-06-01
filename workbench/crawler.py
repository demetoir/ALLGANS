import sys
import time
import random
import traceback
import os
import progressbar
import codecs
import requests
import webbrowser
from bs4 import BeautifulSoup
from util.Logger import StdoutOnlyLogger
from util.misc_util import dump_json, load_json


def deco_exception_handle(func):
    """decorator for catch exception and log"""

    def wrapper(*args, **kwargs):
        self = args[0]
        log_func = self.log
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            log_func("KeyboardInterrupt detected abort process")
        except Exception as e:
            log_error_trace(log_func, e)

    return wrapper


def log_error_trace(log_func, e, head=""):
    exc_type, exc_value, exc_traceback = sys.exc_info()

    msg = '%s\n %s %s : %s \n' % (
        head,
        "".join(traceback.format_tb(exc_traceback)),
        e.__class__.__name__,
        e,
    )
    log_func(msg)


class Crawler:
    def __init__(self):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()
        self.default_save_path = os.path.join('.', 'crawl_result')
        self.result = None

    @deco_exception_handle
    def run(self, urls, n_thread=None, rand_delay=None):
        ret = {}
        for i in progressbar.progressbar(range(len(urls)), redirect_stdout=False):
            url = urls[i]

            ret[url] = self.get_html(url)
            if rand_delay:
                time.sleep(random.uniform(0, 3))
        self.result = ret

        return self.result

    def get_html(self, url):
        _html = ""
        resp = requests.get(url)
        if resp.status_code == 200:
            _html = resp.text
        return _html

    @deco_exception_handle
    def save(self, obj=None, path=None):
        if path is None:
            folder = 'save_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            path = os.path.join(self.default_save_path, folder)
        if obj is None:
            obj = self.result()

        self.check_path(path)

        dump_json(obj, path)

        self.log('save at path={}'.format(path))

    @deco_exception_handle
    def load(self, path):
        return load_json(path)

    def check_path(self, path):
        head, tail = os.path.split(path)
        self.log(head, tail)
        if not os.path.exists(head):
            os.mkdir(head)

    def open_chrome(self, url):
        webbrowser.open(url)


class Soup:
    def __init__(self):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()
