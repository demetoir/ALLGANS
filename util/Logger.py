import logging
import logging.handlers
import os


class Logger:
    """wrapper class of logging module

    # usage
    logger = Logger("i am logger")
    log = logger.get_logger()
    log("logging message")
    log("also", "can use like print",",the build-in function",)
    """
    LOGGER_FORMAT = '%(asctime)s > %(message)s'
    PRINT_LOGGER_FORMAT = '%(asctime)s > %(message)s'

    def __init__(self, name, path=None, file_name='log', level=logging.INFO, stdout_only=False):
        """create logger

        :param name:name of logger
        :param path: log file path
        :param file_name: log file name
        :param level: logging level
        :param stdout_only: default False
        if std_only is True, log message print only stdout not in logfile
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter(Logger.LOGGER_FORMAT)

        self.file_handler = None
        if not stdout_only:
            self.file_handler = logging.FileHandler(os.path.join(path, file_name))
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

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
        def deco(target_function):
            def wrapper(*args):
                return target_function(" ".join(map(str, args)))

            return wrapper

        func = getattr(self.logger, level)
        return deco(func)
