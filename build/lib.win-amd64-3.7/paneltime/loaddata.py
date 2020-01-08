#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions as fu
import date_time
from datetime import datetime
import tempstore

def load(fname,sep,dateformat,load_tmp_data):
	fname=fu.obtain_fname(fname)
	if load_tmp_data:
		data=tempstore.loaddata(fname)
		if not data is None:
			return data
	heading,s=get_head_and_sep(fname,sep)
	print ("opening file ...")
	data=np.loadtxt(fname,delimiter=s,skiprows=1,dtype=bytes)
	data=data.astype(str)
	print ("... done")
	data=convert_to_numeric_dict(data,heading,dateformat)
	tempstore.savedata(fname,data)
	return data

def load_SQL(conn,sql_string,dateformat,load_tmp_data):
	if load_tmp_data:
		data=tempstore.loaddata(sql_string)
		if not data is None:
			return data
	crsr=conn.cursor()
	print ("fetching SQL data ...")
	crsr.execute(sql_string)
	data=np.array(crsr.fetchall())
	print ("... done")
	heading=[]
	dtypes=[]
	for i in crsr.description:
		heading.append(i[0])
		if i[1] in SQL_type_dict:
			dtypes.append(i[1])
		else:
			dtypes.append(None)
	data=convert_to_numeric_dict(data,heading,dateformat,dtypes)
	remove_nan(data)
	tempstore.savedata(sql_string,data)
	return data
	
def remove_nan(data):
	#Todo: add functionality to delete variables that cause too many deletions
	k0=list(data.keys())[0]
	notnan=(np.isnan(data[k0])==0)
	for i in data:
		notnan=(notnan*(np.isnan(data[i])==0))
	for i in data:
		data[i]=data[i][notnan]
	print("%s observations removed because they were nan" %(len(notnan)-np.sum(notnan)))
		


	

	
	
	
def get_name(x,x_names,default):
	x=get_names(x,x_names)
	if x==[]:
		return default
	else:
		return x[0]
	
def get_names(x,x_names):
	if x_names is None:
		if x is None:
			return []
		else:
			x_names=x
	if type(x_names)==list or type(x_names)==tuple or type(x_names)==np.ndarray:
		if type(x_names[0])==str:
			return list(x_names)
		else:
			raise RuntimeError("Variable names need to be string, list or tuples. Type %s cannot be used" %(type(x_names[0])))
	elif type(x_names)==str: 
		if ',' in x_names and '\n' in x_names:
			raise RuntimeError("X-variables needs to be either comma or linfeed separated. You cannot use both")
		for s in [',','\n',' ','\t']:
			if  s in x_names:
				return fu.clean(x_names,s)
			if s=='\t':#happens when used delimiter was not found
				return [fu.clean(x_names)]#allows for the possibilty of a single variable
	else:
		raise RuntimeError("Variable names need to be string, list or tuples. Type %s cannot be used" %(type(x_names)))


def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
	
def convert_to_numeric_dict(a,name,dateformat,dtypes=None):
	N,k=a.shape
	df=dict()
	if dtypes is None:
		dtypes=k*[None]
	for i in range(k):
		make_numeric(a[:,i:i+1],name[i],df,dateformat,dtypes[i])	
	return df
	
def make_numeric(a,name,df,dateformat,dtype):
	if not dtype is None and dtype in SQL_type_dict:
		try:
			df[name]=np.array(a,dtype=SQL_type_dict[dtype])
			return
		except:
			pass
	try:
		try_float_int(a, df, name)
	except ValueError:
		try:
			check_dateness(a,df,name,dateformat)
		except ValueError:
			convert_cat_to_int(a,df,name)
			
def try_float_int(a,df,name):
	a=a.astype(float)
	try:
		if np.all(np.equal(np.mod(a, 1), 0)):
			a=a.astype(int)
	except:
		pass
	df[name]=a	

def convert_cat_to_int(a,df,name):
	print ("""Converting categorical variable %s to integers ...""" %(name,))
	q=np.unique(a)
	d=dict(zip(q, range(len(q))))
	df[name]=np.array([[d[k[0]]] for k in a],dtype=int)
	

def check_dateness(a,df,name,dateformat):
	n,k=a.shape
	dts=np.unique(a)
	d=dict()
	lst=[]
	for dt in dts:
		d[dt]=(datetime.strptime(dt,dateformat)-datetime(1900,1,1)).days
		lst.append(d[dt])
	if np.max(lst)-np.min(lst)<3:#seconds
		for dt in dts:
			d[dt]=(datetime.strptime(dt,dateformat)-datetime(2000,1,1)).seconds	
	df[name]=np.array([[d[k[0]]] for k in a])
	a=0




	
def get_best_sep(string,sep):
	"""Finds the separator that gives the longest array"""
	if not sep is None:
		return sep,string.split(sep)
	sep=''
	maxlen=0
	for i in [';',',',' ','\t']:
		b=head_split(string,i)
		if len(b)>maxlen:
			maxlen=len(b)
			sep=i
			c=b
	return sep,c,maxlen
				
def head_split(string,sep):
	a=string.split(sep)
	b=[]
	for j in a:
		if len(j)>0:
			b.append(j)	
	return b
			
def get_head_and_sep(fname,sep):
	f=open(fname,'r')
	head=f.readline().strip()
	r=[]
	sample_size=20
	for i in range(sample_size):
		r.append(f.read())	
	f.close()
	
	sep,h,n=get_best_sep(head,sep)
	for i in h:
		if is_number(i):
			raise RuntimeError("""The input file must contain a header row. No numbers are allowed in the header row""")
	
	for i in [sep,';',',','\t',' ']:#checks whether the separator is consistent
		err=False
		b=head_split(head, i)
		for j in r:
			if len(j.split(i))!=len(b):
				err=True
				break
			if err:
				h=b
				sep=i
			

	if sep is None:
		raise RuntimeError("Unable to find a suitable seperator for the input file. Check that your input file has identical number of columns in each row")
	return h,sep


def filter_data(filters,data,data_dict):
	"""Filters the data based on setting in the string Filters"""
	if filters is None:
		return data
	fltrs=fu.clean(filters,' and ')
	n=len(data)
	v=np.ones(n,dtype=bool)
	for f in fltrs:
		r=eval(f,globals(),data_dict)
		r.resize(n)
		print ('Removing %s observations due to filter %s' %(np.sum(r==0),f))
		v*=r
	data=data[v]
	k=len(data)
	print ('Removed %s of %s observations - %s observations remaining' %(n-k,n,k))
	return data
				
				
				
				
				
				
SQL_type_dict={0: float,
 1: int,
 2: int,
 3: float,
 4: float,
 5: float,
 6: float,
 8: int,
 9: int,
 16: int,
 246: int
 }