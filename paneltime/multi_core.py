#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import struct
import pickle
import time
import functions as fu
from queue import Queue
from threading import Thread



class master():
	"""creates the slaves"""
	def __init__(self,modules,max_nodes):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """
		if max_nodes is None:
			self.cpu_count=os.cpu_count()#assignment to self allows for the possibility to manipulate the count
		else:
			self.cpu_count=min((os.cpu_count(),max_nodes))
		n=self.cpu_count
		self.slaves=[slave(modules,i) for i in range(n)]
		pids=[self.slaves[i].p_id for i in range(n)]
		info=tuple([n] +pids+[os.getpid()])
		pstr='Multi core processing enabled using %s cores. Slave PIDs: '+(n-1)*'%s, '
		pstr=pstr+'%s. Master PID: %s'
		print (pstr %info)	

	def send_dict(self, d,instructions):
		if instructions !='static dictionary' and instructions !='dynamic dictionary':
			raise RuntimeError('Incorrect instructions passed for adding dictionary to slaves')
		for s in self.slaves:
			s.send(instructions,d)
			res=s.receive()

	def send_holdbacks(self, key_arr):
		"""Sends a list with keys to variables that are not to be returned by the slaves"""
		for s in self.slaves:
			s.send('holdbacks',key_arr)
			res=s.receive()

	def send_tasks(self,tasks):
		"""expressions is a list of (strign,id) tuples with string expressions to be executed. All variables in expressions are stored in the dictionary sent to the slaves"""
		tasks=list(tasks)
		n=len(tasks)
		m=min((self.cpu_count,n))
		d_arr=[]
		for i in range(m):
			self.slaves[i].send('expression evaluation',tasks.pop(0))#initiating the self.cpus first evaluations
		q=Queue()
		for i in range(m):
			t=Thread(target=self.slaves[i].receive,args=(q,),daemon=True)
			t.start()
		got=0
		sent=m
		while 1:
			if got<n:
				r,s=q.get()
				got+=1
				d_arr.append(r)
			if sent<n:
				self.slaves[s].send('expression evaluation',tasks.pop(0))#initiating the self.cpus first evaluations
				t=Thread(target=self.slaves[s].receive,args=(q,),daemon=True)
				t.start()		
				sent+=1
			if sent>=n and got>=n:
				break
		return get_slave_dicts(d_arr)

def get_slave_dicts(d_arr):

	d=d_arr[0]
	for i in range(1,len(d_arr)):
		for key in d_arr[i]:
			if not key in d:
				d[key]=d_arr[i][key]
	return d



class slave():
	"""Creates a slave"""
	command = [sys.executable, "-u", "-m", "slave.py"]


	def __init__(self,modules,slave_id):
		"""Starts local worker"""
		cwdr=os.getcwd()
		os.chdir(__file__.replace(__name__+'.py',''))
		self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		os.chdir(cwdr)
		self.t=transact(self.p.stdout,self.p.stdin)
		self.p_id = self.receive()
		self.slave_id=slave_id
		f_node_name=fu.obtain_output_fname('/output/slaves/%s.txt' %(slave_id,))
		self.send('init_transact',(modules,slave_id,f_node_name))
		pass

	def send(self,msg,obj):
		"""Sends msg and obj to the slave"""
		self.t.send((msg,obj))          

	def receive(self,q=None):

		if q is None:
			answ=self.t.receive()
			return answ
		q.put((self.t.receive(),self.slave_id))


	def kill(self):
		self.send((True,None))





class transact():
	"""Local worker class"""
	def __init__(self,read, write):
		self.r=read
		self.w=write

	def send(self,msg):
		w=getattr(self.w,'buffer',self.w)
		pickle.dump(msg,w)
		w.flush()   

	def send_debug(self,msg,f):
		w=getattr(self.w,'buffer',self.w)
		write(f,str(w))
		pickle.dump(msg,w)
		w.flush()   	

	def receive(self):
		r=getattr(self.r,'buffer',self.r)
		u= pickle.Unpickler(r)
		return u.load()

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()


class multiprocess:
	def __init__(self,max_nodes=None):
		#self.master=master([['regprocs','rp'],['maximize','mx'],['loglikelihood','logl']])#for paralell computing
		self.master=master([],max_nodes)#for paralell computing
		self.d=dict()
		if not self.master is None:
			self.master.send_holdbacks(['AMAp','AMAq','GARM','GARK'])

	def execute(self,expr,run_mp=True):
		"""For submitting multiple functionsargs is an array of argument arrays where the first element in each 
		argument array is the function to be evaluated"""
		if not run_mp:#For debugging purposes
			for i in expr:
				exec(i,None,self.d)#the first element in i is the function, the rest are arguments
		else:
			d=self.master.send_tasks(expr)
			for i in d:
				self.d[i]=d[i]			
		return self.d

	def exe_from_arglist(self,function,args):
		a=[]
		n=len(args)
		for i in range(n):
			f_expr='res%s=' + function
			a.append(f_expr %(i,args[i]))
		self.execute(a)
		res=[]
		return self.d

	def send_dict(self,d,instructions):
		for i in d:
			self.d[i]=d[i]
		if str(type(self.master))=="<class 'multi_core.master'>":
			self.master.send_dict(d,instructions)

def format_args_array(arg_array,run_mp=True):
	for i in range(len(arg_array)):
		arg_array[i]=format_args(arg_array[i], run_mp)
	return arg_array	

def format_args(x,run_mp):
	if not run_mp:
		x=x.replace('rp.','')
	x=x.replace('\t','    ')
	n=0
	xarr=x.split('\n')
	newx=[]
	for j in xarr:
		k=len(j.lstrip())
		if k>0 and len(newx)==0:
			n=len(j)-k
		if k>0:
			newx.append(j[n:])
	newx='\n'.join(newx)
	return newx