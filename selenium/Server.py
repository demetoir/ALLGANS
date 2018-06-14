import os
import subprocess
import time

from util.Logger import StdoutOnlyLogger
from util.misc_util import log_error_trace

JAVA_PATH = 'C:\\Program Files\\Java\\jre-10.0.1\\bin\\java.exe'
SELENIUM_SERVER_JAR_PATH = 'C:\\Users\\demetoir_desktop\\PycharmProjects\\kaggle-MLtools\\selenium\\bin\\selenium-server\\selenium-server-standalone-3.12.0.jar'
CHROME_DRIVER_PATH = os.path.join('.', 'selenium', 'bin', 'chromedriver_win32', 'chromedriver.exe')


class Server:
    def __init__(self, port=4444):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()

        java_path = JAVA_PATH
        selenium_server_jar_path = SELENIUM_SERVER_JAR_PATH
        port_options = " ".join(["-port", str(port)])
        chrome_driver_path = CHROME_DRIVER_PATH
        chrome_driver_option = '-Dwebdriver.chrome.driver={}'.format(chrome_driver_path)
        self.args = " ".join([java_path, chrome_driver_option, '-jar', selenium_server_jar_path, port_options])

        self.server = None

    def open(self):
        if self.server is not None:
            self.log('Remote Selenium server already opened')

        self.server = subprocess.Popen(self.args)

    def close(self):
        if self.server is None:
            self.log('Remote Selenium server already closed')
            return

        self.server.kill()

    def poll(self):
        return self.server.poll()


if __name__ == '__main__':

    # ports = [4444, 4445, 4446, 4447]
    ports = [4444]
    server_address = "http://127.0.0.1:{}/wd/hub"

    servers = [Server(port) for port in ports]
    [server.open() for server in servers]

    try:
        while True:
            time.sleep(1)
            pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stop server")
    except BaseException as e:
        log_error_trace(print, e)
    finally:
        [server.close() for server in servers]
