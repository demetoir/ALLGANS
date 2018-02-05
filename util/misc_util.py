import tarfile
import zipfile

import requests


# pickle util
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


# TODO need relocate
def load_class_from_source_path(module_path, class_name):
    from importlib._bootstrap_external import SourceFileLoader

    module_ = SourceFileLoader('', module_path).load_module()
    return getattr(module_, class_name)


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
