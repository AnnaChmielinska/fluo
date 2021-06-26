====
fluo
====

**fluo** is a set of tools for analysis of fluorescence measurements in
time-domain mode.

:Author: Anna Chmielińska, <anka.chmielinska@gmail.com>.


Installation
============

* Via `pip`::

    python3 -m pip install fluo

* Via `setuptools`::

    python3 setup.py install

fluo requires installation of the following software:

* python >= 3.6
* importlib_metadata = 4.15.0
* tqdm = 4.15.0
* numpy = 1.13.1
* matplotlib = 3.4.2
* scipy = 1.7.0
* numdifftools = 0.9.40
* lmfit = 1.0.2


Usage
=====

**fluo** bundles python objects for construction of models and fitting them to
measured data. Check out workflow examples in `examples` folder. In order to
display graphs with matplotlib you need to install GUI backend (f. e. `tkinter`)
with the OS packaging manger
(more info `here <https://matplotlib.org/stable/tutorials/introductory/usage.html#what-is-a-backend>`_).
