
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import traceback
import datetime
import gc
import transact
import importlib.util
import sys


def session(t,s_id,f):

	d=dict()

	while 1:
		(msg,obj) = t.receive()
		response=None
		if msg=='kill':
			sys.exit()
			response=True
		elif msg == 'dict':#a dictionary to be used in the batch will be passed
			f_dict = open(obj, 'rb')
			u = pickle.Unpickler(f_dict)
			d_new = u.load()
			f_dict.close()
			add_to_dict(d,d_new)
			response = True
		elif msg=='exec':				
			sys.stdout = f
			response = exec(obj,globals(),d)
			sys.stdout = sys.__stdout__
		elif msg=='eval':				
			sys.stdout = f
			response = eval(obj,globals(),d)
			sys.stdout = sys.__stdout__
		elif msg == 'import':
			sys.stdout = f
			d.update(import_module(obj))
			sys.stdout = sys.__stdout__
		else:
			raise RuntimeError('No valid directive supplied')
		t.send(response)		
		gc.collect()
	
	
def add_to_dict(to_dict,from_dict):
	for i in from_dict:
		to_dict[i]=from_dict[i]

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()
	
	
def import_module(path):
	module_name = os.path.splitext(os.path.basename(path))[0]  # Extract the module name from the path
	spec = importlib.util.spec_from_file_location(module_name, path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module  # Optional: Add to sys.modules
	spec.loader.exec_module(module)
	return {'module_name': module}



try: 

	t = transact.Transact(sys.stdin, sys.stdout)
	#Handshake:
	t.send(os.getpid())
	msg,(s_id,fpath) = t.receive()
	#Wait for instructions:
	fname=os.path.join(fpath,'thread %s.txt' %(s_id,))
	f = open(fname, 'w')	
	session(t, s_id, f)
except Exception as e:
	
	f.write('SID: %s      TIME:%s \n' %(s_id,datetime.datetime.now()))
	traceback.print_exc(file=f)
	f.flush()
	f.close()
