"""misc utils
pickle, import module, zip, etc ..."""
from glob import glob
from importlib._bootstrap_external import SourceFileLoader
import tarfile
import zipfile
import requests
import os
import pickle
import types
import json
import sys


def dump_pickle(path, data):
    """dump pickle

    * [warning] use pickle module python3, python2 may incompatible

    :type path: str
    :type data: object
    :param path: dump path
    :param data: target data to dump
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    """load pickle

    * [warning] use pickle module python3, python2 may incompatible

    :type path: str
    :param path: path to load data

    :return: data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def import_class_from_module_path(path, class_name):
    """import class from module path

    :type path: str
    :type class_name: str
    :param path: path of module
    :param class_name: class name to load class from module

    :return: class

    :raise FileNotFoundError
    if path not found
    :raise AttributeError, ImportError
    """
    try:
        module_ = SourceFileLoader('', path).load_module()
        return getattr(module_, class_name)
    except FileNotFoundError:
        raise FileNotFoundError("%s not found" % path)
    except AttributeError:
        raise AttributeError("%s class not found in %s" % (class_name, path))


def module_path_finder(path, name, recursive=True):
    """find module's absolute path in path

    :type path: str
    :type name: str
    :type recursive: bool
    :param path: target path to find module
    :param name: name of module
    :param recursive: option to search path recursively
    default True

    :return: module's absolute path

    :raise FileNotFoundError
    if module file not found in path
    """
    path_ = None
    paths = glob(os.path.join(path, '**', '*.py'), recursive=recursive)
    for path in paths:
        file_name = os.path.basename(path)
        if file_name == name + '.py':
            path_ = os.path.abspath(path)

    if path_ is None:
        raise FileNotFoundError("module %s not found" % name)

    return path_


def imports():
    """iter current loaded modules"""
    for name, val in globals().items():
        if isinstance(val, getattr(types, "ModuleType")):
            yield val.__name__


def dump_json(obj, path):
    """dump json

    :type obj: object
    :type path: str
    :param obj: object to dump
    :param path: path to dump
    """
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path):
    """load json

    :type path: str
    :param path: load path
    """
    with open(path, 'r') as f:
        metadata = json.load(f)
    return metadata


def download_from_url(url, path):
    """download data from url

    :type url: str
    :type path: str
    :param url: download url
    :param path: path to save
    """

    with open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50 - done), done * 2))
                sys.stdout.flush()



def extract_tar(source_path, destination_path):
    """extract tar file

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    """
    with tarfile.open(source_path) as file:
        file.extractall(destination_path)


def extract_zip(source_path, destination_path):
    """extract zip file

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    :return:
    """
    with zipfile.ZipFile(source_path) as file:
        file.extractall(destination_path)


def extract_file(source_path, destination_path):
    """extract file zip, tar, tar.gz, ...

    :type source_path: str
    :type destination_path: str
    :param source_path:source path to extract
    :param destination_path: destination path
    """
    extender_tar = ['tar.gz', 'tar', 'gz']
    extender_zip = ['zip']

    extend = source_path.split('.')[-1]
    if extend in extender_tar:
        extract_tar(source_path, destination_path)
    elif extend in extender_zip:
        extract_zip(source_path, destination_path)
