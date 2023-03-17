#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

version='1.2.12'

from setuptools import setup, find_packages
from codecs import open
from os import path

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
f = open(path.join(here, 'README.txt'), encoding='utf-8')
long_description = f.read()

setup(
    name='paneltime',
    version=version,
    description='An efficient integrated panel and GARCH estimator',
    long_description=long_description,
    url='https://github.com/espensirnes/paneltime',
    author='Espen Sirnes',
    author_email='espen.sirnes@uit.no',
    license='GPL-3.0',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
        ],

  keywords='econometrics',

  packages=find_packages(exclude=['contrib', 'docs', 'tests']),

  install_requires=['numpy >= 1.11','pymysql', 'pandas', 'psutil', 'mpmath'],
	extras_require={'linux':'gcc'},	

  package_data={
      '': ['*.ico','cfunctions/likelihood/ctypes.dll', 'cfunctions/likelihood/ctypes.so'],
      },
  include_package_data=True,

  entry_points={
      'console_scripts': [
          'paneltime=paneltime:main',
          ],
      },
)

