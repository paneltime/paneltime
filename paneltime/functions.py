#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append(__file__.replace("paneltime\\functions.py",'build\\lib.win-amd64-3.5'))
sys.path.append(__file__.replace("paneltime\\functions.py",'build\\lib.linux-x86_64-3.5'))
import cfunctions as c
import numpy as np
import sys
import os
import csv
from scipy import sparse as sp
import re


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
	if type(a)==sp.csc_matrix:
		return a.multiply(b)
	if a is None or b is None:
		return None
	if len(a.shape)==2 and len(b.shape)==2:
		if a.shape[1]!=b.shape[0] and a.shape[0]==b.shape[0]:
			return np.dot(a.T,b)
		return np.dot(a,b)
	if len(a.shape)==2 and len(b.shape)==3:
		N,T,k=b.shape
		x=np.moveaxis(b, 1, 0)
		x=x.reshape((T,N*k))
		x=np.dot(a,x)
		x.resize((T,N,k))
		x=np.moveaxis(x,0,1)
		#slower alternative:
		#x2=np.array([np.dot(a,b[i]) for i in range(b.shape[0])])
		return x
	elif len(a.shape)==3 and len(b.shape)==2:
		return np.array([np.dot(a[i],b) for i in range(a.shape[0])])
	elif len(a.shape)==3 and len(b.shape)==3:
		if a.shape[1]!=b.shape[1]:
			raise RuntimeError("dimensions do not match")
		elif a.shape[0]==b.shape[0] and reduce_dims:
			x=np.sum([np.dot(a[i].T,b[i]) for i in range(a.shape[0])],0)
			return x
		elif a.shape[2]==b.shape[1]:
			k,Ta,Ta2=a.shape
			if Ta2!=Ta:
				raise RuntimeError("hm")
			N,T,m=b.shape
			b_f=np.moveaxis(b, 1, 0)
			a_f=a.reshape((k*T,T))
			b_f=b_f.reshape((T,N*m))
			x=np.dot(a_f,b_f)
			x.resize((k,T,N,m))	
			x=np.swapaxes(x, 2, 0)
			#slower:
			#x2=np.array([[np.dot(a[i],b[j]) for j in range(b.shape[0])] for i in range(a.shape[0])])
			#x2=np.moveaxis(x2,0,2)	
			return x
	elif len(a.shape)==2 and len(b.shape)==4:
		if a.shape[1]!=b.shape[1] or a.shape[1]!=a.shape[0]:
			raise RuntimeError("dimensions do not match")
		else:
			N,T,k,m=b.shape
			x=np.moveaxis(b, 1, 0)
			x=x.reshape((T,N*k*m))
			x=np.dot(a,x)
			x.resize((T,N,k,m))
			x=np.moveaxis(x,0,1)

			#slower alternatives:
			#x=np.array([[np.dot(a,b[i,:,j]) for i in range(b.shape[0])] for j in range(b.shape[2])])
			#x=np.moveaxis(x,0,2)
				#or
			#x2=np.zeros(b.shape)		
			#r=c.dot(a,b,b.shape,x2)		
			return x

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

def split_input(input_str):
	
	if input_str is None:
		return None
	
	p = re.compile('\([^()]+\)')
	if '§' in input_str or '£' in input_str  or '¤' in input_str:
		raise RuntimeError("The charactesr §, ¤  or £ are not allowed in model string")
	while 1:
		matches=tuple(p.finditer(input_str))
		if len(matches)==0:
			break
		for m in matches:
			k,n=m.start(),m.end()
			s =input_str[:k]+'¤'
			s+=input_str[k+1:n-1].replace('+','§') + '£'
			s+=input_str[n:]
			input_str=s

		

	for s in [',','\n','+',' ']:
		lst=input_str.split(s)
		if len(lst)>1:
			break
	x=[]
	for i in lst:
		m=clean(i)
		if m!='':
			m=m.replace('¤','(')
			m=m.replace('§','+')
			m=m.replace('£',')')
			x.append(m)
	return x	


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
	fname=obtain_output_fname(name,extension)
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
		savevar(var,name,extension)

def obtain_output_fname(name,extension=''):
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

def copy_array_dict(d):
	r=dict()
	for i in d:
		r[i]=np.array(d[i])
	return r
