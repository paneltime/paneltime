
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import datetime
import pickle
import gc
import inspect
import numpy as np
from threading import Thread

def main(t,initcommand,s_id,fpath):
	fname=os.path.join(fpath,'slaves/%s.txt' %(s_id,))
	f=open(fname,'w',1)
	d=dict()
	exec(initcommand,globals(),d)
	d_list=list(d.keys())
	holdbacks=[]
	callback_class = CallBack(t, f)
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
			fobj,expr=obj
			ftr=open(fobj, "rb")
			u= pickle.Unpickler(ftr)
			d_new=u.load()		
			add_to_dict(d,d_new)
			d_list=list(d.keys())
			response=True
			ftr.close()
			exec(expr,globals(),d)
		elif msg=='exec':				
			sys.stdout = f
			exec(obj,globals(),d)
			sys.stdout = sys.__stdout__
			response=release_dict(d,d_list,holdbacks)
		elif msg=='listen':			
			sys.stdout = f
			d['callback'] = callback_class.callback
			try:
				exec(obj,globals(),d)
			except Exception as e:
				if str(e)!='Quit from callback':
					raise RuntimeError(e)
				else:
					write(f, 'quit after callback')
			sys.stdout = sys.__stdout__
			t.send((callback_class.d, True))
		elif msg=='holdbacks':
			holdbacks.extend(obj)
		if not msg in ['listen']:
			t.send(response)		
		if  msg=='exec':#remove response dict after sending
			for i in response:
				d.pop(i)
		gc.collect()
		

class CallBack:
	def __init__(self, t, f):
		self.t = t
		self.f = f
		
	def write(self, s):
		write(self.f, s)
	
	def callback(self, d={}):
		t.send((d,False))
		msg, (self.d, quit) = t.receive()
		if quit:
			raise RuntimeError('Quit from callback')
		return self.d
	
	
	
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
		if (not i in d_list) and (not i in holdbacks) and (not inspect.ismodule(d[i])):
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