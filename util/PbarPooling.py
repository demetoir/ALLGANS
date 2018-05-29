from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.Logger import StdoutOnlyLogger
import queue


class PbarPooling:
    def __init__(self, func=None, n_parallel=4):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()
        self.func = func
        self.n_parallel = n_parallel
        self.pool = Pool(self.n_parallel)

    def run(self, jobs=None, func=None, n_parallel=None):
        if func is None:
            func = self.func
        if n_parallel is None:
            n_parallel = self.n_parallel

        self.log('start pooling queue {} jobs'.format(len(jobs)))

        ret = [None] * len(jobs)

        with tqdm(total=len(jobs)) as bar:
            procs = [self.pool.apply_async(func, job) for job in jobs]

            q = queue.Queue()
            for idx, proc in enumerate(procs):
                q.put((proc, idx))

            while q.qsize() > 0:
                proc, idx = q.get()
                try:
                    ret[idx] = proc.get(timeout=0.01)
                    bar.update(1)
                except BaseException:
                    q.put((proc, idx))

        self.log('end pooling queue')
        return ret
