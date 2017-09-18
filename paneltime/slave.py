
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import numpy as np
import regprocs as rp
import maximize as mx
import loglikelihood as logl
import time

def main(f):
	t=multi_core.transact(sys.stdin, sys.stdout)
	t.send(os.getpid())
	msg,(modules,s_id,f_node_name)=t.receive()
	f_node=open(f_node_name,'w')
	aliases=[]
	for module,alias in modules:
		if alias=='':
			exec('import '+module)
			aliases.append(module)
		else:
			exec('import '+module +' as ' + alias)
			aliases.append(alias)
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
			for a in aliases:
				d[a]=vars()[a]
			d_old=dict(d)
			response=True
		elif msg=='expression evaluation':	
			sys.stdout = f_node
			exec(obj,globals(),d)
			sys.stdout = sys.__stdout__
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
	write(f, 'test')
	traceback.print_exc(file=f)
	f.flush()
	f.close()
	sys.exit()