#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions as fu
import date_time
from datetime import datetime
from datetime import date
import tempstore
NON_NUMERIC_TAG='|~|'
forbidden_names=['tobit_low','tobit_high','Intercept']

def load(fname,sep,dateformat,load_tmp_data):
	fname=fu.obtain_fname(fname)
	if load_tmp_data:
		data=tempstore.loaddata(fname)
		if not data is None:
			load_data_printout(data)
			return data
	heading,s=get_head_and_sep(fname,sep)
	print ("opening file ...")
	data=np.loadtxt(fname,delimiter=s,skiprows=1,dtype=bytes)
	data=data.astype(str)
	print ("... done")
	data=convert_to_numeric_dict(data,heading,dateformat)
	tempstore.savedata(fname,data)
	load_data_printout(data)
	return data

def load_json(fname):
	file=open(fname,encoding='utf-8')
	data=json.load(file)
	df=dict()	
	for r in data:
		for k in r:
			if type(r[k])==dict():
				for m in r[k]:
					key=k+'_'+m
					append(df, key, r[k][m])
			else:
				append(df, k, r[k])
	return df
						
def append(d,key,i):
	if key in d:
		d[key].append(i)
	else:
		d[key]=[i]

def load_SQL(conn,sql_string,dateformat,load_tmp_data):
	if load_tmp_data:
		data=tempstore.loaddata(sql_string)
		if not data is None:
			print('local data loaded')
			load_data_printout(data)
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
	load_data_printout(data)
	return data

def load_data_printout(data):
	lst=[]
	for i in data:
		if not NON_NUMERIC_TAG in i:
			lst.append(i)
	print ("The following variables were loaded:"+', '.join(lst))
	
def remove_nan(data):
	#Todo: add functionality to delete variables that cause too many deletions
	lst=list(data.keys())
	
	for k in lst:
		try:
			notnan=(np.isnan(data[k])==0)
			break
		except:
			pass
	for i in data:
		if not NON_NUMERIC_TAG in i:
			try:
				isnan=np.isnan(data[i])
				if not sum(isnan)>int(0.5*len(data[i])):
					notnan=(notnan*(isnan==0))
			except:
				pass
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
	
def convert_to_numeric_dict(data,names,dateformat,dtypes=None):
	N,k=data.shape
	df=dict()
	if dtypes is None:
		dtypes=k*[None]
	for i in range(k):
		name=names[i]
		if name in forbidden_names:
			print(f"You can't call a variable {name}, since it is in use by paneltime. The variable is renamed {name}_" )
			name=name+'_'
		make_numeric(data[:,i:i+1],name,df,dateformat,dtypes[i])	
	return df
	
def make_numeric(variable,name,df,dateformat,dtype):
	if not dtype is None and dtype in SQL_type_dict:
		try:
			df[name]=np.array(variable,dtype=SQL_type_dict[dtype])
			return
		except:
			pass
	try:
		try_float_int(variable, df, name)
	except:
		try:
			check_dateness(variable,df,name,dateformat)
		except ValueError:
			convert_cat_to_int(variable,df,name)
		df[name+NON_NUMERIC_TAG]=variable#adds the original to the df
			
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
	a[a==np.array(None)]=''
	dts=np.unique(a)
	d=dict()
	lst=[]
	
	if 'datetime.date' in str(type(dts[0])):
		for dt in dts:
			d[dt]=(dt-date(1900,1,1)).days
			lst.append(d[dt])
	else:
		for dt in dts:
			d[dt]=(datetime.strptime(dt,dateformat)-datetime(1900,1,1)).days
			lst.append(d[dt])
	if np.max(lst)-np.min(lst)<3:#seconds
		if 'datetime.date' in str(type(dts[0])):
			for dt in dts:
				d[dt]=(dt-date(2000,1,1)).seconds
				lst.append(d[dt])
		else:
			for dt in dts:
				d[dt]=(datetime.strptime(dt,dateformat)-datetime(2000,1,1)).seconds
				lst.append(d[dt])
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