import random
import time


class Process:
    def __init__(self):
        self.call_count = 0
        self.url = None
        self.job_idx = None

    def task(self, url=None, job_idx=None):
        if url is None:
            url = self.url

        if job_idx is None:
            job_idx = self.job_idx

        self.call_count += 1
        # print('sleep', job_idx)
        print('url={}, job_idx={}'.format(str(url), job_idx))

        sleep_time = random.uniform(0, 2)
        print('sleep {}, {}'.format(sleep_time, job_idx))
        time.sleep(sleep_time)
        print('wake', job_idx)

        return job_idx
