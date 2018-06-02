import os
from selenium import webdriver


class Chrome:
    def __init__(self, headless=True, driver_path=None):
        self.driver = None

        args = webdriver.ChromeOptions()

        if headless:
            args.add_argument('headless')
        args.add_argument("--disable-gpu")
        args.add_argument("disable-gpu")

        if driver_path is None:
            driver_path = os.path.join('.', 'selenium', 'bin', 'chromedriver_win32', 'chromedriver')

        self.driver = webdriver.Chrome(driver_path, chrome_options=args)


class RemoteChrome:
    def __init__(self, host='http://127.0.0.1', port=4444, headless=True):
        self.driver = None
        args = webdriver.ChromeOptions()

        if headless:
            args.add_argument('headless')

        args.add_argument("--disable-gpu")
        args.add_argument("disable-gpu")

        self.driver = webdriver.Remote(
            command_executor='{host}:{port}/wd/hub'.format(host=host, port=str(port)),
            desired_capabilities=args.to_capabilities())
