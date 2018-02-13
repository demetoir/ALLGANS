"""misc util
pickle, import module, zip, etc ..."""
import tarfile
import zipfile
import requests
import os
from glob import glob
from importlib._bootstrap_external import SourceFileLoader


def dump(path, data):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    del pickle


def load(path):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    del pickle
    return data


def import_class_from_module_path(module_path, class_name):
    try:
        module_ = SourceFileLoader('', module_path).load_module()
        return getattr(module_, class_name)
    except AttributeError:
        raise ImportError("%s class not found in %s" % (class_name, module_path))


def module_path_finder(module_path, module_name, recursive=True):
    path_ = None
    paths = glob(os.path.join(module_path, '**', '*.py'), recursive=recursive)
    for path in paths:
        file_name = os.path.basename(path)
        if file_name == module_name + '.py':
            path_ = path

    if path_ is None:
        raise ModuleNotFoundError("module %s not found" % module_name)

    return path_


def imports():
    import types
    for name, val in globals().items():
        if isinstance(val, getattr(types, "ModuleType")):
            yield val.__name__


def download_data(source_url, download_path):
    r = requests.get(source_url, allow_redirects=True)
    open(download_path, 'wb').write(r.content)


def extract_tar(source_path, destination_path):
    with tarfile.open(source_path) as file:
        file.extractall(destination_path)


def extract_zip(source_path, destination_path):
    with zipfile.ZipFile(source_path) as file:
        file.extractall(destination_path)


def extract_data(source_path, destination_path):
    extender_tar = ['tar.gz', 'tar', 'gz']
    extender_zip = ['zip']

    extend = source_path.split('.')[-1]
    if extend in extender_tar:
        extract_tar(source_path, destination_path)
    elif extend in extender_zip:
        extract_zip(source_path, destination_path)
