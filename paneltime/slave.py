
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import numpy as np
import time

def main(f):
	t=multi_core.transact(sys.stdin, sys.stdout)
	t.send(os.getpid())
	msg,(modules,s_id,f_node_name)=t.receive()
	f_node=open(f_node_name,'w')
	aliases=[]
	d_init=dict()
	for module,alias in modules:
		if alias=='':
			exec('import '+module,globals(),d_init)
			aliases.append(module,globals(),d_init)
		else:
			exec('import '+module +' as ' + alias,globals(),d_init)

	holdbacks=[]
	while 1:
		(msg,obj) = t.receive()
		response=None
		if msg==True:
			sys.exit()
			response=True
		elif msg=='static dictionary':#an initial dictionary to be used in the batch will be passed
			d=obj
			add_to_dict(d_init,d)
			response=True
		elif msg=='dynamic dictionary':#a dictionary to be used in the batch will be passed
			d=obj
			add_to_dict(d,d_init)
			d_list=list(d.keys())
			response=True
		elif msg=='expression evaluation':	
			sys.stdout = f_node
			exec(obj,globals(),d)
			sys.stdout = sys.__stdout__
			response=release_dict(d,d_list,holdbacks)
		elif msg=='holdbacks':
			holdbacks=obj  
			
		t.send(response)
		
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
	f=open('slave_errors.txt','w')
	main(f)
except Exception as e:
	traceback.print_exc(file=f)
	f.flush()
	f.close()
	sys.exit()