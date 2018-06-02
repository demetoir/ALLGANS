import codecs
import time
import os
import requests
from util.deco import deco_exception_handle
from util.misc_util import dump_json, load_json, check_path


class Crawler:
    def __init__(self, save_path=None):
        super().__init__()

        self.logger = None
        self.log = print
        if save_path is None:
            save_path = '.'
        self.save_path = os.path.join(save_path, 'crawl_result',
                                      'save_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        self.result = None
        self.url = None
        self.job_idx = None

    @deco_exception_handle
    def run(self, urls, n_parallel=4, rand_delay=None):
        pass

    def task(self, url=None, job_idx=None):
        if url is None:
            url = self.url
        if job_idx is None:
            job_idx = self.job_idx

        html = self.get_html(url)
        self.save_html(html, path_tail=str(job_idx) + '.html')
        return job_idx

    def get_html(self, url):
        _html = ""
        resp = requests.get(url)
        if resp.status_code == 200:
            # resp.encoding = None
            _html = resp.text
            # _html = _html.encode(resp.encoding, errors='ignore').decode('utf8', error='ignore')
        return _html

    def save_json(self, obj=None, path=None):
        if path is None:
            folder = 'save_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            path = os.path.join(self.save_path, folder)
        if obj is None:
            obj = self.result()

        dump_json(obj, path)

        self.log('save at path={}'.format(path))

    @staticmethod
    def load_json(path):
        return load_json(path)

    def save_html(self, html, path_head=None, path_tail=None, encoding='UTF8'):
        if path_head is None:
            path_head = self.save_path

        check_path(path_head)
        path = os.path.join(path_head, path_tail)
        with codecs.open(path, 'w', encoding=encoding) as f:
            f.writelines(html)
        self.log('save at {}'.format(path))

    def save_lines(self, lines, path_head=None, path_tail=None, encoding='UTF8'):
        if path_head is None:
            path_head = self.save_path

        new_lines = []
        for line in lines:
            line = str(line)
            if line[-1] is not '\n':
                new_lines += [line + '\n']
            else:
                new_lines += [line]
        merged_line = "".join(new_lines)

        check_path(path_head)
        path = os.path.join(path_head, path_tail)
        with codecs.open(path, 'w', encoding=encoding) as f:
            f.writelines(merged_line)
        self.log('save at {}'.format(path))
