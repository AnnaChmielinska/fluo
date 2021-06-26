#!/usr/bin/env python3

from setuptools import setup, find_packages
from os.path import join, dirname
import re

version = re.search("__version__ = '([^']+)'",
                    open('./fluo/__init__.py').read()).group(1)

setup(
    name='fluo',
    version=version,
    author='Anna Chmielinska',
    author_email='anka.chmielinska@gmail.com',
    url = 'https://github.com/AnnaChmielinska/fluo',
    download_url = 'https://github.com/AnnaChmielinska/fluo/archive/{}.tar.gz'.format(version),
    license='General Public License 3',
    description='Fluorescence in time domain toolkit.',
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    packages = find_packages(),
    install_requires=[
        'matplotlib >= 3.4.2, < 3.4.3',
        'lmfit >= 1.0.2, < 1.0.3',
        'numdifftools >= 0.9.40, < 0.9.41',
        'scipy >= 1.7.0, < 1.7.1',
        'numpy >= 1.21.0, < 1.21.1',
        'tqdm >= 4.15.0, < 4.15.1',
        'importlib_metadata >= 4.5.0, < 4.5.1'
    ]
)
