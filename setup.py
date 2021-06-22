#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

version='1.1.13'

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path
import subprocess
import sys
import time

try:
	import numpy as np
except:
	try:
		subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
		time.sleep(2)
		import numpy as np
	except:
		print("""Warning: the python package numpy is not preinstalled. 
Without it the 'cfunctions' library will not be installed, which will affect performance.
Try uninstall paneltime, install numpy and install paneltime again""")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.txt'), encoding='utf-8') as f:
	long_description = f.read()
try:
	ext=[]
	cfunc=Extension('cfunctions',sources=['paneltime/cfunctions.cpp'],include_dirs=[np.get_include()])
	ext=[cfunc]
except:
	print("""Warning: 'cfunctions' library not compiled. This may affect performance. 
If you are installing on a Windows machine, precompiled versions exist for python 3.5,3.6 and 3.7, so you should be fine""")
setup(
    name='paneltime',
    ext_modules=ext,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='An efficient integrated panel and GARCH estimator',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/espensirnes/paneltime',

    # Author details
    author='Espen Sirnes',
    author_email='espen.sirnes@uit.no',

    # Choose your license
    license='GPL-3.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Statistical Software :: time series estimation',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-3.0 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        ],

    # What does your project relate to?
    keywords='econometrics',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    
    
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #**************************************************************************REMOVED>
    install_requires=['numpy >= 1.11','scipy','matplotlib','pymysql'],
    #**************************************************************************<REMOVED

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #**************************************************************************REMOVED>
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    #     },
    #**************************************************************************<REMOVED 
    #
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #**************************************************************************REMOVED>
    
    package_data={
        '': ['gui/*', '*.ico','gui/img/*.png',
		'cfunctions.cp35-win_amd64.pyd',
		'cfunctions.cp36-win_amd64.pyd',
		'cfunctions.cpython-35m-x86_64-linux-gnu.so',
		'cfunctions.cp37-win_amd64.pyd'],
        },
    include_package_data=True,
    #**************************************************************************<REMOVED
    #
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #**************************************************************************REMOVED>
    #data_files=[('my_data', ['data/data_file'])],
    #**************************************************************************<REMOVED
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'paneltime=paneltime:main',
            ],
        },
)