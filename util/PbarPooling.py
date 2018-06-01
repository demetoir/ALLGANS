import os
from multiprocessing import Pool
from tqdm import tqdm
from util.Logger import Logger
from util.deco import log_error_trace
from util.misc_util import dump_pickle, dump_json, time_stamp


class PbarPooling:
    def __init__(self, func=None, n_parallel=4, initializer=None, initargs=()):
        self.logger = Logger(self.__class__.__name__)
        self.log = self.logger.get_log()
        self.func = func
        self.n_parallel = n_parallel
        self.pool = Pool(self.n_parallel, initializer=initializer, initargs=initargs)
        self.procs = None
        self.pbar = None

    def map(self, func=None, jobs=None):
        if func is None:
            func = self.func

        self.log('start pooling queue {} jobs'.format(len(jobs)))

        self.pbar = tqdm(total=len(jobs))

        def update_qbar(*args):
            self.pbar.update(1)

        self.procs = []
        for job in jobs:
            proc = self.pool.apply_async(func, job, callback=update_qbar)

            self.procs += [(proc, job)]

        ret = self.get()

        self.log('end pooling queue')
        return ret

    def map_async(self, func=None, jobs=None):
        if func is None:
            func = self.func

        self.log('start pooling queue {} jobs'.format(len(jobs)))

        self.bar = tqdm(total=len(jobs))

        def update_qbar(args):
            self.bar.update(1)

        self.procs = [self.pool.apply_async(func, job, callback=update_qbar) for job in jobs]

    def get(self):
        if self.procs is None:
            return None

        rets = []
        self.fail_list = []

        for proc, job in self.procs:
            try:
                ret = proc.get()
            except BaseException as e:
                log_error_trace(self.log, e)
                self.log("job fail, proc={proc}, job={job}".format(proc=str(proc), job=str(job)))
                self.fail_list += [job]
                ret = None

            rets += [ret]

        self.pbar.close()
        self.pbar = None
        return rets

    def save_fail_list(self, path=None):
        if path is None:
            path = os.path.join('.', 'fail_list', time_stamp())

        dump_pickle(self.fail_list, path)
        dump_json(list(map(str, self.fail_list)), path)
