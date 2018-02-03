from setuptools import setup
from setuptools import find_packages
from env_settting import *

import io


def readme_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


def folder_init(description="description", default_path=""):
    print(description)
    path = default_path
    print("[default path=%s]" % path)
    while True:
        print("((y)es/(n)o)", end='')
        user_input = input()
        if user_input == 'y' or user_input == 'yes':
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
            break
        elif user_input == 'n' or user_input == 'no':
            print("input path=", end="")
            path = input()
            print("is path=%s ?" % path)
        else:
            print("abort setup")
            exit()
    print()


install_requires_no_tensorflow_gpu = [
    "numpy",
    "tensorflow>=1.4.1",
    "requests>=2.18.4",
    "opencv-python>=3.4.0.12",
    "scikit-learn>=0.19.1",
    "scikit-image>=0.13.1",
    "pillow>=5.0.0",
    "pandas>=0.22.0",
    "matplotlib>=2.1.2",
]

install_requires_with_tensorflow_gpu = [
    "numpy",
    "tensorflow>=1.4.1",
    "tensorflow-gpu>=1.4.1",
    "requests>=2.18.4",
    "opencv-python>=3.4.0.12",
    "scikit-learn>=0.19.1",
    "scikit-image>=0.13.1",
    "pillow>=5.0.0",
    "pandas>=0.22.0",
    "matplotlib>=2.1.2",
]

description = '머신러닝 프레임워크를 활용한 비교사(Unsupervised) 학습 모델 구현 프로젝트'

is_install_tensorflow_gpu = False


print(description)
# init instance folder

instance_dir_setup_description = "setup instance directory"
folder_init(instance_dir_setup_description, os.path.join(ROOT_PATH, 'instance'))

# init data folder
data_dir_setup_description = "setup data directory"
folder_init(data_dir_setup_description, os.path.join(ROOT_PATH, 'data'))

# setup tensorflow gpu version
description = "after installing tensorflow gpu version, you may need to install NVIDIA CUDA \n" \
              "if you want more information of installing NVIDIA CUDA for tesnorflow gpu version go to link bellow\n" \
              "https://www.tensorflow.org/install/ \n" \
              "do you want to install tensorflow gpu? version\n"

print(description)

print("((y)es/(n)o)", end='')
user_input = input()
if user_input == 'y' or user_input == 'yes':
    is_install_tensorflow_gpu = True
elif user_input == 'n' or user_input == 'no':
    is_install_tensorflow_gpu = False
else:
    print("abort setup")
    exit()

if is_install_tensorflow_gpu:
    install_requires = install_requires_with_tensorflow_gpu
else:
    install_requires = install_requires_no_tensorflow_gpu

setup(name='ALLGANS',
      version='0.1',
      description=description,
      long_description=readme_description(),
      url='https://github.com/stunstunstun/awesome-algorithms',
      author='demetoir, WKBae, StarG',
      author_email='wnsqlehlswk@naver.com, williambae1@gmail.com, psk7142@naver.com',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      classifiers=[
          'Programming Language :: Python :: 3.6',
      ],
      zip_safe=False,
      )
