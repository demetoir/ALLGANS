import codecs
import sys
import time
import traceback


def deco_file_lines_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
    def wrapper():
        with codecs.open(in_file, 'r', encoding=encoding) as f:
            lines = [line for line in f.readlines()]

        new_line = []
        for line in lines:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            new_line += [line]
        lines = new_line

        print("read '{}', {} lines".format(in_file, len(lines)))

        lines = func(lines)

        if lines is not None:
            with codecs.open(out_file, 'w', encoding=encoding) as f:
                f.writelines(lines)
            print("write '{}', {} lines".format(out_file, len(lines)))

    wrapper.__name__ = func.__name__
    return wrapper


def deco_file_str_job(func, in_file='input.txt', out_file='output.txt', encoding='UTF8'):
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


def log_error_trace(log_func, e, head=""):
    exc_type, exc_value, exc_traceback = sys.exc_info()

    msg = '%s\n %s %s : %s \n' % (
        head,
        "".join(traceback.format_tb(exc_traceback)),
        e.__class__.__name__,
        e,
    )
    log_func(msg)


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
        print("total time {}".format(time.time() - start))

        return ret

    wrapper.__name__ = func.__name__
    return wrapper
