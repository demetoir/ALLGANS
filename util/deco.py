import codecs
import time

from util.misc_util import log_error_trace


def file_lines_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
    def wrapper():
        read_time = time.time()
        with codecs.open(in_file, 'r', encoding=encoding) as f:
            lines = [line for line in f.readlines()]

        new_lines = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            new_lines += [line]
        lines = new_lines

        read_time = time.time() - read_time
        old_len = len(lines)
        print("read '{}', {} lines, {:.3f}'s elapsed".format(in_file, old_len, read_time))

        func_time = time.time()
        lines = func(lines)
        func_time = time.time() - func_time
        print("in func {:.3f}'s elapsed".format(func_time))

        write_time = time.time()

        if lines is not None:
            new_lines = []
            for line in lines:
                line = str(line)
                if line[-1] is not '\n':
                    new_lines += [line + '\n']
                else:
                    new_lines += [line]
            lines = new_lines

            with codecs.open(out_file, 'w', encoding=encoding) as f:
                f.writelines(lines)
            write_time = time.time() - write_time
            new_len = len(lines)

            if old_len - new_len == 0:
                print('same len')
            elif old_len - new_len > 0:
                print("del {} lines".format(old_len - new_len))
            else:
                print("add {} lines".format(-(old_len - new_len)))

            print("write '{}', {} lines, {:.3f}'s elapsed".format(out_file, new_len, write_time))
        else:
            write_time = 0
        print("total {:.4f}'s elapsed".format(read_time + func_time + write_time))

    wrapper.__name__ = func.__name__
    return wrapper


def file_str_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
    def wrapper():
        with codecs.open(in_file, 'r', encoding=encoding) as f:
            line = "".join([line for line in f.readlines()])

        print("read '{}', {} length".format(in_file, len(line)))

        line = func(line)

        if line is not None:
            with codecs.open(out_file, 'w', encoding=encoding) as f:
                f.writelines(str(line))
            print("write '{}', {} length".format(out_file, len(line)))

    wrapper.__name__ = func.__name__
    return wrapper


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

    wrapper.__name__ = func.__name__
    return wrapper


def deco_log_func_name(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        log_func = self.log
        return log_func(func.__name__, *args, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


def deco_timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print("time {:.3f}'s elapsed".format(time.time() - start))

        return ret

    wrapper.__name__ = func.__name__
    return wrapper
