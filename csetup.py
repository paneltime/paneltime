#!/usr/bin/env python
import sys
sys.argv.append('build')
from distutils.core import setup, Extension
import posix


cfunctions = Extension('cfunctions',
#        define_macros=[('FOO', '1')],
		include_dirs=[],
		library_dirs=[],
		libraries=[],
		sources=['cdef.cpp'])

setup(name='Stocks',
		version='0.1',
		description='Stocks',
		author='Espen Sirnes',
		author_email='espen.sirnes@chem.uit.no',
		url='http://depot.uit.no/projects/stocks',
		long_description="""
        No description available.
        """,
		ext_modules=[cfunctions])

aaa=1