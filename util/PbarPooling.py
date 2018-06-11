import os
import signal
from multiprocessing import Pool
from queue import Queue
from tqdm import tqdm
from util.Logger import Logger
from util.misc_util import dump_pickle, dump_json, time_stamp, log_error_trace


def init_worker(*args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PbarPooling:

    def __init__(self, func=None, n_parallel=4, initializer=None, initargs=(), child_timeout=30):
        self.logger = Logger(self.__class__.__name__)
        self.log = self.logger.get_log()

        self.func = func

        self.n_parallel = n_parallel

        if initializer is None:
            self.initializer = init_worker
        else:
            self.initializer = initializer
        self.initargs = initargs
        self.child_timeout = child_timeout

        self.pools = [Pool(1, initializer=init_worker, initargs=initargs) for _ in range(n_parallel)]
        self.queues = [Queue() for _ in range(n_parallel)]
        self.pbar = None
        self.fail_list = []

    def map(self, func=None, jobs=None):
        self.map_async(func, jobs)
        return self.get()

    def map_async(self, func=None, jobs=None):
        if func is not None:
            self.func = func

        self.log('start pooling queue {} jobs'.format(len(jobs)))

        self.pbar = tqdm(total=len(jobs))

        def update_pbar(args):
            self.pbar.update(1)

        self.update_pbar = update_pbar

        self.jobs = jobs
        for i in range(len(jobs)):
            pool_id = i % self.n_parallel
            job = jobs[i]
            pool = self.pools[pool_id]
            child = pool.apply_async(func, job, callback=update_pbar)
            self.queues[pool_id].put((child, job))

    def get(self):
        rets = []
        while sum([q.qsize() for q in self.queues]) > 0:
            for pool_id in range(self.n_parallel):
                ret = None
                if self.queues[pool_id].qsize() == 0:
                    continue
                child, job = self.queues[pool_id].get()
                try:
                    ret = child.get(timeout=self.child_timeout)
                except KeyboardInterrupt:
                    self.log("KeyboardInterrupt terminate pools\n"
                             "{fail}/{total} fail".format(fail=len(self.fail_list), total=len(self.jobs)))
                    self.terminate()
                    raise KeyboardInterrupt
                except BaseException as e:
                    log_error_trace(self.log, e)
                    self.log("job fail, kill job={job}, child={child}".format(child=str(None), job=str(job[3])))
                    self.pbar.update(1)
                    self.fail_list += [job]
                    self.pools[pool_id].terminate()
                    self.pools[pool_id].join()
                    self.pools[pool_id] = Pool(1, initializer=self.initializer, initargs=self.initargs)

                    new_queue = Queue()
                    while self.queues[pool_id].qsize() > 0:
                        _, job = self.queues[pool_id].get()
                        child = self.pools[pool_id].apply_async(self.func, job, callback=self.update_pbar)
                        new_queue.put((child, job))

                    self.queues[pool_id] = new_queue

                finally:
                    rets += [ret]

        self.pbar.close()
        self.log("{fail}/{total} fail".format(fail=len(self.fail_list), total=len(self.jobs)))
        self.log('end pooling queue')
        return rets

    def save_fail_list(self, path=None):
        if path is None:
            path = os.path.join('.', 'fail_list', time_stamp())

        dump_pickle(self.fail_list, path + ".pkl")
        dump_json(list(map(str, self.fail_list)), path + ".json")

    def terminate(self):
        for pool_id in range(self.n_parallel):
            self.pools[pool_id].terminate()
            self.pools[pool_id].join()

        if self.pbar is not None:
            self.pbar.close()

    def close(self):
        for pool_id in range(self.n_parallel):
            self.pools[pool_id].close()
            self.pools[pool_id].close()
            self.pools[pool_id].join()

        if self.pbar is not None:
            self.pbar.close()
