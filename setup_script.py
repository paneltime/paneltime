#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os

def main():
	nukedir('dist')
	nukedir('build')
	nukedir('paneltime.egg-info')
	os.system('python setup.py bdist_wheel sdist build')
	
	
def rm(fldr):
	try:
		shutil.rmtree(fldr)
	except Exception as e:
		print(e)

def nukedir(dir):
	if dir[-1] == os.sep: dir = dir[:-1]
	if not os.path.isfile(dir):
		return
	files = os.listdir(dir)
	for file in files:
		if file == '.' or file == '..': continue
		path = dir + os.sep + file
		if os.path.isdir(path):
			nukedir(path)
		else:
			os.unlink(path)
	os.rmdir(dir)
	
main()