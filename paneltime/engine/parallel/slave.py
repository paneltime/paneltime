
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import traceback
import datetime
import time
import gc
import transact
import sys
import importlib

import importlib.util
import importlib.machinery


class Session:
	def __init__(self, t, s_id, f):

		self.d=dict()
		while 1:
			(msg,obj) = t.receive()
			response=None
			if msg=='kill':
				sys.exit()
				response=True
			elif msg == 'dict':#a dictionary to be used in the batch will be passed
				self.dict(obj)
			elif msg=='exec':				
				self.exec(f, obj)
			elif msg=='eval':				
				response = self.eval(f, obj)		
			else:
				raise RuntimeError('No valid directive supplied')
			t.send(response)		
			gc.collect()
	
	
	def dict(self, obj):
		f_dict = open(obj, 'rb')
		u = pickle.Unpickler(f_dict)
		d_new = u.load()
		f_dict.close()
		add_to_dict(self.d,d_new)
		
	def exec(self, f, obj):
		sys.stdout = f
		t = time.time()
		exec(obj,globals(),self.d)
		print(f'Procedure: {obj} \nTime used: {time.time()-t}')
		sys.stdout = sys.__stdout__	

	
	def eval(self, f, obj):
		
		sys.stdout = f
		response = eval(obj,globals(),self.d)
		sys.stdout = sys.__stdout__
		return response	


	
def add_to_dict(to_dict,from_dict):
	for i in from_dict:
		to_dict[i]=from_dict[i]

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()
	

try: 

	t = transact.Transact(sys.stdin, sys.stdout)
	#Handshake:
	t.send(os.getpid())
	msg,(s_id,fpath) = t.receive()
	#Wait for instructions:
	fname=os.path.join(fpath,'thread %s.txt' %(s_id,))
	f = open(fname, 'w')	
	Session(t, s_id, f)
except Exception as e:
	
	f.write('SID: %s      TIME:%s \n' %(s_id,datetime.datetime.now()))
	traceback.print_exc(file=f)

	f.flush()
	f.close()
