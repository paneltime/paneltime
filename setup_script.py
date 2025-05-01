#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os
import re
import subprocess as sp
import sys
import glob
import psutil
from paneltime import opt_module

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def main():

	push_git = '-g' in sys.argv
	push_pip = '-p' in sys.argv
	addver = not ('-k' in sys.argv)


	opt_module.options_to_txt()

	nukedir(f'{CUR_DIR}/dist')
	nukedir(f'{CUR_DIR}/build')
	nukedir(f'{CUR_DIR}/paneltime.egg-info')
	remove_pycache_dirs(CUR_DIR)
	create_readme()

	wd = os.path.dirname(__file__)
	os.chdir(wd)
	if push_git or push_pip:
		version = add_version(wd, addver)
		print(f"Incrementet to version {version}")

	if push_git:
		gitpush(version)
	else:
		print('Not pushed to git - use "-g" to push to git')
	
	os.system('python -m build')

	if push_pip:
		os.system("twine upload dist/*")
	else:
		print('Not pushed to pypi - use "-p" to push to pypi (pip)')
	


def gitpush(version):
	print(f"Packaging paneltime version {version}")
	r = sp.check_output('git pull')
	if r != b'Already up to date.\n':
		raise RuntimeError(f'Not up to date after git pull. Fix any conflicts and check that the repository is up to date\nPull output:\n{r})')
	os.system('git add .')
	os.system(f'git commit -m "New version {version} committed: {input("Write reason for commit (without quotation marks): ")}"')
	os.system('git push')	
	
def add_version(wd, add=True):
	srchtrm = r"(\d+\.\d+\.\d+)"
	version = re_replace('pyproject.toml', srchtrm, wd, add=add)
	re_replace('index.md', srchtrm, wd, version)
	re_replace('paneltime/info.py', srchtrm, wd, version)
	return version

def re_replace(fname, searchterm, wd, version = None, add=True):
	fname = os.path.join(wd, fname)
	f = open(fname, 'r')
	s = f.read()
	m = re.search(searchterm, s, re.MULTILINE)
	if version is None:
		v = s[m.start(0):m.end(0)]
		v = v.split('.')
		v = v[0], v[1], str(int(v[2])+add)
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
	try:
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
		return
	except (FileNotFoundError, PermissionError):
		return

def create_readme():
	src=f"{CUR_DIR}/index.md"
	dest=f"{CUR_DIR}/README.md"
	with open(src, "r", encoding="utf-8") as f:
		lines = f.readlines()

	if lines[0].strip() == "---":
		# Skip until closing '---'
		end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
		content = lines[end+1:]
	else:
		content = lines

	with open(dest, "w", encoding="utf-8") as f:
		f.writelines(content)


def remove_pycache_dirs(root):
	for dirpath, dirnames, filenames in os.walk(root):
		if '__pycache__' in dirnames:
			pycache_path = os.path.join(dirpath, '__pycache__')
			print(f"Removing {pycache_path}")
			nukedir(pycache_path)


main()