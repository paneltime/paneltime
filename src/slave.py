
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import numpy as np

def main(f):
	t=multi_core.transact(sys.stdin, sys.stdout)
	t.send(os.getpid())
	[module,alias],s_id=t.receive()
	if alias=='':
		exec('import '+module)
	else:
		exec('import '+module +' as ' + alias)
		module=alias
	d_init=dict()
	holdbacks=[]
	while 1:
		(msg,obj) = t.receive()
		response=None
		if msg==True:
			sys.exit()
			response=True
		elif msg=='static dictionary':#an initial dictionary to be used in the batch will be passed
			d_init=obj
			response=True
		elif msg=='dynamic dictionary':#a dictionary to be used in the batch will be passed
			d=obj
			for i in d_init:
				d[i]=d_init[i]
			d[module]=vars()[module]
			d_old=dict(d)
			response=True
		elif msg=='expression evaluation':
			exec(obj,globals(),d)
			response=release_dict(d,d_old,holdbacks)
		elif msg=='holdbacks':
			holdbacks=obj                     
		t.send(response)
		
def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()
	
def release_dict(d,d_old,holdbacks):
	"""Ensures that only new variables are returned"""
	response=dict()
	for i in d:
		if (not i in d_old) and (not i in holdbacks):
			response[i]=d[i]	
	return response

try: 
	f=open('slave_errors.txt','w')
	main(f)
except Exception as e:
	traceback.print_exc(file=f)
	f.flush()
	f.close()
	sys.exit()