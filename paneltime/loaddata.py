#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions as fu
import date_time

def load(fname,sep):
	fname=fu.obtain_output_fname(fname,'csv')
	heading,s=get_head_and_sep(fname,sep)
	print ("opening file ...")
	data=np.loadtxt(fname,delimiter=s,skiprows=1,dtype=np.str)
	print ("... done")
	data=np.char.rstrip(np.char.lstrip(data,"b'"),"'")
	data=convert_to_numeric(data,heading)
	d=dict()
	for i in range(data.shape[1]):
		d[heading[i]]=data[:,i:i+1]
	return d

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
	
def convert_to_numeric(r_old,name):
	N,k=r_old.shape
	r=None
	for i in range(k):
		r=make_numeric(r_old[:,i:i+1],name[i],r)	
	return r
	
def make_numeric(a,name,r):
	try:
		b=np.array(a,dtype=np.float64)
	except ValueError:	
		d=np.unique(a)
		dt,is_datetime=extract_date_time(d)
		if is_datetime:
			print ("""Converting categorical variable %s to days since 2000 ...""" %(name,))
			d=dict(zip(d, dt.flatten()))
		else:
			print ("""Converting categorical variable %s to integers ...""" %(name,))
			d=dict(zip(d, range(len(d))))
		a=a.flatten()
		b=np.array([d[k] for k in a],dtype=float)
		b=b.reshape(len(b),1)
	if r is None:
		r=b
	else:
		r=np.append(r,b,1)
	return r

def extract_date_time(d):
	n=len(d)
	r=np.zeros((n,7),dtype=np.float)
	OK=True
	for i in range(len(d)):
		dt_str=d[i]
		v=dt_str.split()
		r[i][0:3],date_OK=extract_date(v[0])
		r[i][3:7],time_OK=extract_time(v[len(v)==2])
		if date_OK==False and time_OK==False:
			OK=False
			return d,False#Abort if both time and date parsing fails
	hours,minutes,seconds=analyze_time(r)
	years,days,months=analyze_date(r)
	days_since=date_time.days_since(years,months,days,hours,minutes,seconds,2000)
	return days_since,True

def analyze_time(r):
	if np.sum(r[:,3:7])==0:
		return 0,0,0
	hours=r[:,3:4]
	minutes=r[:,4:5]
	seconds=r[:,5:6]
	am_pm==r[:,6:7]
	if np.sum(am_pm)>0:
		hours=hours*(am_pm==1)+(12+hours)*(am_pm==2)
	if np.max(hours)>24.0 or np.max(minutes)>60.0 or np.max(seconds)<0 or  np.min(months)<0:
		raise RuntimeError("There is something wrong with a string variable that is formatted like a date-time variable")
	return hours,minutes,seconds
	

def analyze_date(r):
	if np.sum(r[:,0:3])==0:
		return 0,0,0	
	x1=None
	for i in range(3):
		if np.max(r[:,i:i+1])>31:#this is the year
			years=r[:,i:i+1]
		else:
			if x1 is None:
				x1=r[:,i:i+1]
			else:
				x2=r[:,i:i+1]

	if np.max(x1)>12:#this is the day
		days=x1
		months=x2
	else:
		days=x2
		months=x1

	return years,days, months
			
		
def extract_date(dt):
	if '-' in dt:
		d=dt.split('-')
	elif '/' in dt:
		d=dt.split('/')	
	else:
		return (0,0,0),False
	if len(d)!=3:
		return (0,0,0),False
	return d,True

	
def extract_time(t):
	if not ':' in t:
		return (0,0,0,0),False
	v=t.split(':')
	if len(v)!=3:
		return (0,0,0,0),False
	am_pm=0
	if v[2][-2].lower()=='am': 
		am_pm=1
	elif v[2][-2].lower()=='pm':
		am_pm=2
	v[2]=v[2][:2]
	return v,True
	
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
				
				
				
				
				
				
			
			
			