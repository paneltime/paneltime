
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle
import traceback
import datetime
import gc
import transact


def session(t,s_id,fpath):
	fname=os.path.join(fpath,'slaves/%s.txt' %(s_id,))
	f=open(fname,'w',1)
	d=dict()

	while 1:
		(msg,obj) = t.receive()
		response=None
		if msg=='kill':
			sys.exit()
			response=True
		elif msg == 'dict':#a dictionary to be used in the batch will be passed
			f = open(obj, 'rb')
			u = pickle.Unpickler(f)
			d_new = u.load()
			f.close()
			add_to_dict(d,d_new)
			response = True
		elif msg=='exec':				
			sys.stdout = f
			response = eval(obj,globals(),d)
			sys.stdout = sys.__stdout__
		elif msg=='eval':				
			sys.stdout = f
			response = eval(obj,globals(),d)
			print(response)
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


try: 

	t = transact.Transact(sys.stdin, sys.stdout)
	#Handshake:
	t.send(os.getpid())
	msg,(s_id,fpath) = t.receive()

	fname = 'slave_errors.txt'
	f = open(fname,'w')
	f.write('kglkgg')

	f.flush()
	f.close()

	#Wait for instructions:
	session(t,s_id,fpath)
except Exception as e:
	f.write('SID: %s      TIME:%s \n' %(s_id,datetime.datetime.now()))
	traceback.print_exc(file=f)
	f.flush()
	f.close()
	raise RuntimeError(e)