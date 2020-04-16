#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import struct
import pickle
import datetime
from queue import Queue
from threading import Thread
import time



class master():
	"""creates the slaves"""
	def __init__(self,initcommand,max_nodes, holdbacks,tempfile):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """
		f=tempfile.TemporaryFile()
		self.f=f.name
		f.close()
		if max_nodes is None:
			self.cpu_count=os.cpu_count()#assignment to self allows for the possibility to manipulate the count
		else:
			self.cpu_count=max_nodes
		n=self.cpu_count
		fpath=obtain_fname('./output/')
		self.slaves=[slave(initcommand,i,fpath) for i in range(n)]
		pids=[]
		for i in range(n):
			pid=str(self.slaves[i].p_id)
			if int(i/5.0)==i/5.0:
				pid='\n'+pid
			pids.append(pid)
		self.send_holdbacks(holdbacks)
		pstr="""Multi core processing enabled using %s cores. \n
Master PID: %s \n
Slave PIDs: %s"""  %(n,os.getpid(),', '.join(pids))
		print (pstr)

	def send_dict(self, d,cpu_ids=None):
		if cpu_ids is None:
			cpu_ids=range(self.cpu_count)
		for i in cpu_ids:
			self.slaves[i].send('dictionary',d)
			res=self.slaves[i].receive()

	def send_dict_by_file(self, d,cpu_ids=None):
		f=open(self.f,'wb')
		pickle.dump(d,f)   
		f.flush() 
		f.close()
		if cpu_ids is None:
			cpu_ids=range(self.cpu_count)
		for i in cpu_ids:
			self.slaves[i].send('filetransfer',self.f)
		for i in cpu_ids:
			res=self.slaves[i].receive()
			
			
	def remote_recieve(self, variable,node):
		"""Sends a list with keys to variables that are not to be returned by the slaves"""
		self.slaves[node].send('remote recieve',variable)
		return self.slaves[node].receive()

	def send_holdbacks(self, key_arr):
		"""Sends a list with keys to variables that are not to be returned by the slaves"""
		if key_arr is None:
			return
		for s in self.slaves:
			s.send('holdbacks',key_arr)
			res=s.receive()

	def send_tasks(self,tasks,remote=False,timer=False,progress_bar=None):
		"""tasks is a list of (strign,id) tuples with string expressions to be executed. All variables in expressions are stored in the dictionary sent to the slaves
		if remote=True, the list must be a list of tuples, where the first itmes is the expression and the second is the variable that should be returned"""
		tasks=list(tasks)
		n=len(tasks)
		m=min((self.cpu_count,n))
		d_arr=[]
		if timer:
			t_arr=[-time.perf_counter()]*n
		if not remote:
			msg='expression evaluation'
		else:
			msg='remote expression valuation'		
		for i in range(m):
			self.slaves[i].send(msg,tasks.pop(0))#initiating the self.cpus first evaluations
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
				if timer:
					t_arr[s]+=time.perf_counter()
			if sent<n:
				self.slaves[s].send(msg,tasks.pop(0))#supplying additional tasks for returned cpus
				t=Thread(target=self.slaves[s].receive,args=(q,),daemon=True)
				t.start()		
				sent+=1
			if sent>=n and got>=n:
				break
			if not progress_bar is None:
				pb_func,pb_min,pb_max,text=progress_bar			
				pb_func(pb_min+(pb_max-pb_min)*got/n,text)
		d=get_slave_dicts(d_arr)
		if timer:
			return d,t_arr
		return d
	
	def quit(self):
		for i in self.slaves:
			i.p.stdout.close()
			i.p.stderr.close()
			i.p.stdin.close()
			i.p.kill()
			i.p.wait()

def get_slave_dicts(d_arr):

	d=d_arr[0]
	for i in range(1,len(d_arr)):
		for key in d_arr[i]:
			if key in d:
				raise RuntimeWarning('Slaves returned identical variable names. Some variables will be over-written')
			d[key]=d_arr[i][key]
	return d



class slave():
	"""Creates a slave"""
	command = [sys.executable, "-u", "-m", "multi_core_slave.py"]


	def __init__(self,initcommand,slave_id,fpath):
		"""Starts local worker"""
		cwdr=os.getcwd()
		os.chdir(os.path.dirname(__file__))
		self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		os.chdir(cwdr)
		self.t=transact(self.p.stdout,self.p.stdin)
		self.p_id = self.receive()
		self.slave_id=slave_id
		self.send('init_transact',(initcommand,slave_id,fpath))
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
		self.p.kill()





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
		try:
			return u.load()
		except EOFError as e:
			if e.args[0]=='Ran out of input':
				raise RuntimeError("""An error occured in one of the spawned sub-processes. 
Check the output in "slave_errors.txt' in your working directory or 
run without multiprocessing\n %s""" %(datetime.datetime.now()))
			else:
				raise RuntimeError('EOFError:'+e.args[0])

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()


class multiprocess:
	def __init__(self,tempfile,max_nodes=None,initcommand='',holdbacks=None):
		self.d=dict()
		self.master=master(initcommand,max_nodes,holdbacks,tempfile)#for paralell computing



	def execute(self,expr,timer=False,progress_bar=None):
		"""For submitting multiple functionsargs is an array of argument arrays where the first element in each 
		argument array is the function to be evaluated"""
		d=self.master.send_tasks(expr,timer=timer,progress_bar=progress_bar)
		if timer:
			d,t=d
		for i in d:
			self.d[i]=d[i]
		if timer:
			return self.d,t
		return self.d
	
	def remote_execute(self,expr):
		"""For submitting multiple functionsargs is an array of argument arrays where the first element in each 
		argument array is the function to be evaluated"""
		d=self.master.send_tasks(expr,True)
		return d

	def remote_recieve(self,variable,node):
		"""Fetches variable from node after remote_execute"""
		ret=self.master.remote_recieve(variable,node)
		return ret
		
	
	def send_dict_by_file(self,d,cpu_ids=None):
		for i in d:
			self.d[i]=d[i]		
		self.master.send_dict_by_file(d,cpu_ids)

	def exe_from_arglist(self,function,args):
		a=[]
		n=len(args)
		for i in range(n):
			f_expr='res%s=' + function
			a.append(f_expr %(i,args[i]))
		self.execute(a)
		return self.d

	def send_dict(self,d,cpu_ids=None):
		for i in d:
			self.d[i]=d[i]
		if 'multi_core.master' in str(type(self.master)):
			self.master.send_dict(d,cpu_ids)
			
	def quit(self):
		self.master.quit()


def obtain_fname(name):

	path=os.path.abspath(name)
	path_dir=os.path.dirname(path)
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)	

	return path