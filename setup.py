#!/usr/bin/env python

from distutils.core import setup

setup(
    name='jiang2016',
    version='0.3',
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    description="Schemata for the analysis of the data recorded by Xiaolong and Shan. ",
    packages=['jiang2016'],
    requires=['numpy', 'pymysql', 'matplotlib','datajoint', 'commons'],
    license = "MIT",
)
