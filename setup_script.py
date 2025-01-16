#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os
import re
import subprocess as sp
import sys
import glob

def main():

	push_git = '-g' in sys.argv
	push_pip = '-p' in sys.argv
	try:
		nukedir('dist')
		nukedir('build')
		nukedir('paneltime.egg-info')
	except FileNotFoundError:
		pass
	wd = os.path.dirname(__file__)
	os.chdir(wd)

	version = add_version(wd)
	print(f"Incrementet to version {version}")

	if push_git:
		gitpush(version)
	else:
		print('Not pushed to git - use "-g" to push to git')
	
	os.system('python setup.py bdist_wheel sdist build')

	if push_pip:
		os.system("twine upload dist/*")
	else:
		print('Not pushed to pypi - use "-g" to push to pypi (pip)')
	


def gitpush(version):
	print(f"Packaging paneltime version {version}")
	r = sp.check_output('git pull')
	if r != b'Already up to date.\n':
		raise RuntimeError(f'Not up to date after git pull. Fix any conflicts and check that the repository is up to date\nPull output:\n{r})')
	os.system('git add .')
	os.system(f'git commit -m "New version {version} committed: {input("Write reason for commit: ")}"')
	os.system('git push')	
	
def add_version(wd):
	version = re_replace('setup.py', "(?<=version=')(.*)(?=')", wd)
	re_replace('README.md', "(?<=Version:\s)([^\s]+)", wd, version)
	re_replace('paneltime/info.py', "(?<=version=')(.*)(?=')", wd, version)
	return version

def re_replace(fname, searchterm, wd, version = None):
	fname = os.path.join(wd, fname)
	f = open(fname, 'r')
	s = f.read()
	m = re.search(searchterm, s)
	if version is None:
		v = s[m.start(0):m.end(0)]
		v = v.split('.')
		v = v[0], v[1], str(int(v[2])+1)
		version = '.'.join(v)
	s = s[:m.start(0)] +  version + s[m.end(0):]
	tmpname = fname.replace('.', '~.')
	save(tmpname, s)
	save(fname,s)
	os.remove(tmpname)

	return version

	
def save(file, string):
	f = open(file,'w')
	f.write(string)
	f.close()
	
	
	
	
def rm(fldr):
	try:
		shutil.rmtree(fldr)
	except Exception as e:
		print(e)

def nukedir(dir):
	if dir[-1] == os.sep: dir = dir[:-1]
	if os.path.isfile(dir):
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