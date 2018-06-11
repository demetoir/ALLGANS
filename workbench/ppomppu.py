import glob
import os
import re
import shutil
import time
import tqdm
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from util.Crawler import Crawler
from util.PbarPooling import PbarPooling
from util.deco import deco_timeit, file_lines_job
from util.misc_util import check_path, print_lines, log_error_trace


def url_query_parser(url):
    url, query = url.split('?')
    query_list = query.split('&')

    query_dict = {}
    for s in query_list:
        name, val = s.split('=')
        query_dict[name] = val

    return query_dict


@deco_timeit
@file_lines_job
def ppompu_url_finder(lines):
    lines = None

    path = "E:\\crawl_result\\crawl_result\\ppomppu_list"
    files = sorted(glob.glob(path + '\\*.html', recursive=True))
    print("{} files loaded".format(len(files)))

    regex = re.compile("""/zboard/[a-zA-Z]*.php\?id=[a-zA-Z]*&no=[0-9]*""")

    ret = []
    for file in tqdm.tqdm(files):
        with open(file, mode='r', encoding='UTF8') as f:
            soup = BeautifulSoup(f, 'lxml')

        for item in soup.find_all('a'):
            href = str(item.get('href'))
            if regex.match(href):
                ret += [href]

    return ret


def execute_js_go_comment_page(driver, query, index):
    base = """javascript:go_page( "{id}", {no}, {idx}); return false;"""
    script = base.format(id=query['id'], no=query['no'], idx=index)
    driver.execute_script(script)

    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CLASS_NAME, "pagelist_han"))
    )


def execute_js_show_comment_page(driver, query):
    base = """javascript:go_page( "{id}", {no}, 1); return false;"""
    script = base.format(id=query['id'], no=query['no'])
    driver.execute_script(script)

    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CLASS_NAME, "pagelist_han"))
    )


def get_max_comment_page(soup):
    try:
        res = soup.find('font', {'class': 'pagelist_han'})
        item = [int(item.text) for item in res.find_all('a') if item.text.isnumeric()]
        return max(item)
    except BaseException as e:
        return -1


def pomppu_singlepage_crawl(driver, crawler, url, job_id):
    try:
        # time.sleep(10)
        query = url_query_parser(url)
        driver.get(url)

        try:
            if driver.find_element_by_class_name("pre"):
                execute_js_show_comment_page(driver, query)
        except BaseException as e:
            pass

        str_soup = []

        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')

        # main content
        main_content_args = ('td', {'class': "board-contents"})
        for item in soup.find_all(*main_content_args):
            str_soup += [item.text]

        # comment
        comment_args = ('textarea', {'class': 'ori_comment'})
        for item in soup.find_all(*comment_args):
            str_soup += [item.text]

        try:
            max_page = get_max_comment_page(soup)
            # print('{} max_page {}'.format(job_id, max_page))
            for i in range(2, max_page + 1):
                execute_js_go_comment_page(driver, query, i)
                element = driver.find_element_by_id("quote")

                part = str(element.get_attribute('innerHTML'))
                part = BeautifulSoup(part, 'lxml')

                for soup in part.find_all(*comment_args):
                    str_soup += [str(soup.text)]
        except BaseException:
            pass

        str_soup = "\n".join(str_soup)
        crawler.save_html(str_soup, path_tail=str(job_id) + ".txt")
    except BaseException as e:
        log_error_trace(print, e)


def dummy_task(driver, crawler, url, job_id):
    time.sleep(5)
    print('{} done'.format(job_id))
    return None


@deco_timeit
@file_lines_job
def pomppu_page_crawl(lines, n_parallel=4):
    n_parallel = 4
    crawler = Crawler(save_path='E:\\crawl_result')

    bucket_size = 512
    for b in range(0, len(lines), bucket_size):
        print('open driver')
        ports = [4444, 4445, 4446, 4447]
        # drivers = [RemoteChrome(port=4444).driver for _ in range(4)]

        jobs = []
        for i in range(b, min(b + bucket_size, len(lines))):
            job_id = i % n_parallel
            # driver = drivers[job_id]
            driver = None
            crawler = crawler
            url = lines[i]
            id_ = i
            jobs += [(driver, crawler, url, id_)]

        pooling = PbarPooling(n_parallel=n_parallel, func=pomppu_singlepage_crawl, child_timeout=2)
        try:
            # pooling.map(func=pomppu_singlepage_crawl, jobs=jobs)
            pooling.map(func=dummy_task, jobs=jobs)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            break
        except BaseException as e:
            log_error_trace(print, e)
        finally:
            # [driver.quit() for driver in drivers]
            pooling.save_fail_list()
            print('close drivers')


def merge_folder(in_path, out_path):
    # output_path = "E:\\crawl_result\\crawl_result\\merged"
    pattern = """E:\\crawl_result\\crawl_result\save*"""
    folders = list(sorted(glob.glob(pattern)))

    sub_files = {}
    for folder in folders:
        files = glob.glob(str(folder) + "\\*")
        files = sorted(files)
        sub_files[str(folder)] = files

    # pprint(sub_files)

    for key in sub_files:
        new_files = []
        files = sub_files[key]
        for file in files:
            head, tail = os.path.split(file)
            tail = tail.replace(".txt", "")
            new_files += [(int(tail), file)]

        sub_files[key] = sorted(new_files)

    check_path(out_path)

    file_idx = 0
    for key in sub_files:
        files = sub_files[key]
        for i, path in files:
            file_idx += 1
            file_name = os.path.join(out_path, str(file_idx).zfill(5) + '.txt')
            print('from {}\n to {}'.format(path, file_name))
            shutil.copy(path, file_name)

    print('job_done')


def filter_line(line, black_list):
    for black in black_list:
        line = line.replace(black, "")
    return line


def filter_files():
    root_path = "E:\\crawl_result\\crawl_result\\merged"
    check_path(root_path)
    out_path = "E:\\crawl_result\\crawl_result\\output"
    check_path(out_path)

    files = glob.glob(root_path + '\\*')

    max_line = len(files)
    # max_line = 100
    files = files[:max_line]
    for file in files:
        lines = []
        head, tail = os.path.split(file)
        with open(file, mode='r', encoding='UTF8') as f:
            for line in f.readlines():
                lines += [line]

        outs = []
        for line in lines:
            black_list = ["\n", "\r", "<b>", "</b>", 'ㆍ']
            line = filter_line(line, black_list)

            # filter white space
            line = " ".join(line.split())

            # filter html tag
            line = re.sub("<[^>]*>", "", line)

            # filter len 0
            if len(line) == 0:
                continue
            outs += [line]

        new_path = os.path.join(out_path, tail)
        with open(new_path, mode='w', encoding='UTF8') as f:
            for line in outs:
                f.write(line + '\n')

        print_lines(outs)
