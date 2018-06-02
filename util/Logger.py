import logging
import logging.handlers
import os

from util.misc_util import time_stamp, check_path


class Logger:
    """wrapper class of logging module

    # usage
    logger = Logger("i am logger")
    log = logger.get_logger()
    log("logging message")
    log("also", "can use like print",",the build-in function",)
    """
    FILE_LOGGER_FORMAT = '[%(levelname)s] %(asctime)s> %(message)s'
    PRINT_LOGGER_FORMAT = '%(asctime)s> %(message)s'
    NO_FORMAT = ""

    def __init__(self, name, path=None, file_name=None, level=logging.INFO, with_file=True, no_format=False):
        """create logger

        :param name:name of logger
        :param path: log file path
        :param file_name: log file name
        :param level: logging level
        :param with_file: default False
        if std_only is True, log message print only stdout not in logfile
        """
        self.logger = logging.getLogger(name + time_stamp())
        self.logger.setLevel(level)

        self.file_handler = None
        if with_file:
            if path is None:
                path = os.path.join('.', 'log')
            if file_name is None:
                file_name = time_stamp()

            check_path(path)
            self.file_handler = logging.FileHandler(os.path.join(path, file_name))
            self.file_handler.setFormatter(logging.Formatter(self.FILE_LOGGER_FORMAT))
            self.logger.addHandler(self.file_handler)

        if no_format:
            format_ = self.NO_FORMAT
        else:
            format_ = self.PRINT_LOGGER_FORMAT
        formatter = logging.Formatter(format_)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(formatter)
        self.logger.addHandler(self.stream_handler)

    def __repr__(self):
        return self.__class__.__name__

    def __del__(self):
        if self.file_handler is not None:
            self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.stream_handler)
        del self.logger

    def get_log(self, level='info'):
        """return logging function

        :param level: level of logging
        :return: log function
        """

        # catch *args and make to str
        def deco(func):
            def wrapper(*args):
                return func(" ".join(map(str, args)))

            wrapper.__name__ = func.__name__
            return wrapper

        func = getattr(self.logger, level)
        return deco(func)


class StdoutOnlyLogger(Logger):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__

        super().__init__(name, with_file=False, no_format=True)


"""
self.logger = Logger(self.__class__.__name__)
self.log = self.logger.get_log()

self.logger = StdoutOnlyLogger(self.__class__.__name__)
self.log = self.logger.get_log()
"""
