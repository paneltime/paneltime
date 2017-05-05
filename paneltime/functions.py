#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import csv


def timer(tic, a):
	if a is None:
		a=[]
		tic=time.clock()
	tac=time.clock()
	a.append(tic-tac)
	return tac,a
	
def dot(a,b,reduce_dims=True):
	"""returns the dot product of a*b where either a or be or both to be
	arrays of matrices"""
	if a is None or b is None:
		return None
	if len(a.shape)==2 and len(b.shape)==2:
		if a.shape[1]!=b.shape[0] and a.shape[0]==b.shape[0]:
			return np.dot(a.T,b)
		return np.dot(a,b)
	if len(a.shape)==2 and len(b.shape)==3:
		return np.array([np.dot(a,b[i]) for i in range(b.shape[0])])
	elif len(a.shape)==3 and len(b.shape)==2:
		return np.array([np.dot(a[i],b) for i in range(a.shape[0])])
	elif len(a.shape)==3 and len(b.shape)==3:
		if a.shape[1]!=b.shape[1]:
			raise RuntimeError("dimensions do not match")
		elif a.shape[0]==b.shape[0] and reduce_dims:
			return np.sum([np.dot(a[i].T,b[i]) for i in range(a.shape[0])],0)
		elif a.shape[2]==b.shape[1]:
			x=np.array([[np.dot(a[i],b[j]) for j in range(b.shape[0])] for i in range(a.shape[0])])
			return np.moveaxis(x,0,2)	
	elif len(a.shape)==2 and len(b.shape)==4:
		if a.shape[1]!=b.shape[1] or a.shape[1]!=a.shape[0]:
			raise RuntimeError("dimensions do not match")
		else:
			x=np.array([[np.dot(a,b[i,:,j]) for i in range(b.shape[0])] for j in range(b.shape[2])])
			return np.moveaxis(x,0,2)	
	else:
		raise RuntimeError("this multiplication is not supported by dot")


def clean(string,split='',cleanchrs=['\n','\t',' ']):
	"""Cleans the text for linfeed etc., and splits the text wiht split if split is not None. \n
	If return_string a string is returned when the lenght of the split string is 1"""
	if split is None or split=='':
		s=clean_str(string,cleanchrs)
		return s		
	if any([i in split for i in cleanchrs]):
		s=string.split(split)
		for i in range(len(s)):
			s[i]=clean_str(s[i],cleanchrs)
	else:	
		s=clean_str(string,cleanchrs,split)
		
	ret=[]
	for i in s:
		if i!='':
			ret.append(i)
	return ret


def clean_str(s,cleanchrs,split=''):
	for j in cleanchrs:
		s=s.replace(j,'')
	if split!='':
		s=s.split(split)
	return s
	
	
def exec_strip(exestr,glob,loc):
	"""Identical to exec, except leading spaces/tabs are stripped in order to avoid indentation error"""
	if exestr is None:
		return
	lines=exestr.split('\n')
	if len(lines)==1:
		exec(exestr, glob, loc)
	k=len(exestr)
	i=0
	while i<len(lines):
		s=lines[i]
		if len(s.lstrip())>0:
			k=min((s.find(s.lstrip()),k))
			i+=1
		else:
			lines.pop(i)
			
	if k==0 or k==len(exestr):
		exec(exestr, glob, loc)
	r=''
	for s in lines:
		if lines[0][:k]!=s[:k]:
			raise RuntimeError("It appears that the indentation is not uniform for the string. It must either be tabs only or spaces only")
	for s in lines:
		r=r+s[k:]+'\n'
	exec(r, glob, loc)
	
	
def replace_many(string,oldtext_list,newtext):
	for i in oldtext_list:
		string=string.replace(i,newtext)
		
		
def savevar(variable,name='tmp',extension=''):
	"""takes variable and name and saves variable with filname <name>.csv """	
	fname=obtain_fname(name,extension)
	print ( 'saves to '+ fname)
	if type(variable)==np.ndarray:
		if not variable.dtype=='float64':
			savelist(variable, fname)	
		else:
			np.savetxt(fname,variable,delimiter=";")
	else:
		savelist(variable, fname)
		
		
def savelist(variable,name):
	file = open(name,'w',newline='')
	writer = csv.writer(file,delimiter=';')
	writer.writerows(variable)
	file.close()	

def savevars(varlist,extension=''):
	"""takes a tuple of (var,name) pairs and saves numpy array var 
	with <name>.csv. Use double brackets for single variable."""	

	for var,name in varlist:
		savevar(variable,name,extension)

def obtain_fname(name,extension=''):
	name=name.replace('\\','/')
	wd=os.getcwd().replace('\\','/')
	wd_arr=wd.split('/')

	if not '/' in name:
		name='output/'+name
	d_arr=name.split('/')
	d=[]
	for i in d_arr:
		if len(i)>0:
			d.append(i)
	if d[0]=='.':
		d=d[1:]#ignoring single dot, as it is assumed that './'='/' 
	if '.' in d[0]:
		k=len(d[0].split('.'))-2
		if k<len(wd_arr):
			d=d[1:]
			wd_arr=wd_arr[:-k]
			wd='/'.join(wd_arr)	
	else:
		match=0
		for i in range(len(d)-1):#matching wd with file dir
			n=len(wd_arr)
			for j in range(len(wd_arr)):
				if d[i]==wd_arr[j] and match==0:
					match=j
				if match>0 and d[i+j-match]!=wd_arr[j]:
					match=0
					break
			if match>0:
				d=d[i+n-match:]
				break
		
	fname=d[-1]
	if extension!='':
		fname=fname.replace('.'+extension,'')+'.'+extension
	d=wd+'/'+'/'.join(d[:-1])
	if not os.path.exists(d):
		os.makedirs(d)	
	fname=d+'/'+fname	
	return fname