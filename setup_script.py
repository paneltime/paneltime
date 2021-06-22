#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os

def main():
	rm('dist')
	rm('build')
	rm('paneltime.egg-info')
	os.system('python setup.py bdist_wheel sdist build')
	
	
def rm(fldr):
	try:
		shutil.rmtree('dist')
	except:
		pass


main()