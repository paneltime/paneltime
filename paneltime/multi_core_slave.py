
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import datetime
import pickle
import gc
import numpy as np

def main(t,initcommand,s_id,fpath):
	fname=os.path.join(fpath,'slaves/%s.txt' %(s_id,))
	f=open(fname,'w',1)
	d=dict()
	exec(initcommand,globals(),d)
	d_list=list(d.keys())
	holdbacks=[]
	while 1:
		(msg,obj) = t.receive()
		response=None
		if msg=='kill':
			sys.exit()
			response=True
		elif msg=='dictionary':#a dictionary to be used in the batch will be passed
			d_new=obj
			add_to_dict(d,d_new)
			d_list=list(d.keys())
			response=True
		elif msg=='filetransfer':
			ftr=open(obj, "rb")
			u= pickle.Unpickler(ftr)
			d_new=u.load()		
			add_to_dict(d,d_new)
			d_list=list(d.keys())
			response=True
			ftr.close()
		elif msg=='remote expression valuation':
			expr,ret_var=obj
			sys.stdout = f
			exec(expr,globals(),d)
			sys.stdout = sys.__stdout__	
			response={ret_var:d[ret_var]}
		elif msg=='remote recieve':
			response=d[obj]
		elif msg=='expression evaluation':	
			sys.stdout = f
			exec(obj,globals(),d)
			sys.stdout = sys.__stdout__
			response=release_dict(d,d_list,holdbacks)	
		elif msg=='holdbacks':
			holdbacks.extend(obj)		
		t.send(response)		
		if  msg=='expression evaluation':#remove response dict after sending
			for i in response:
				d.pop(i)
		gc.collect()
		

def add_to_dict(to_dict,from_dict):
	for i in from_dict:
		to_dict[i]=from_dict[i]

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()

def release_dict(d,d_list,holdbacks):
	"""Ensures that only new variables are returned"""
	response=dict()
	for i in d:
		if (not i in d_list) and (not i in holdbacks):
			response[i]=d[i]	
	return response

try: 

	t=multi_core.transact(sys.stdin, sys.stdout)
	#Handshake:
	t.send(os.getpid())
	msg,(initcommand,s_id,fpath)=t.receive()
	fname=os.path.join(fpath,'slave_errors.txt')
	f=open(fname,'w')
	#Wait for instructions:
	main(t,initcommand,s_id,fpath)
except Exception as e:
	f.write('SID: %s      TIME:%s \n' %(s_id,datetime.datetime.now()))
	traceback.print_exc(file=f)
	f.flush()
	f.close()
	raise RuntimeError(e)