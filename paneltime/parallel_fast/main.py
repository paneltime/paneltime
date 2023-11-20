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
import loglikelihood as lgl
import tempfile


class master():
	"""creates the slaves"""
	def __init__(self,initcommand,max_nodes, holdbacks):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """
		self.tempfile=tempfile
		
		if max_nodes is None:
			self.cpu_count=os.cpu_count()#assignment to self allows for the possibility to manipulate the count
		else:
			self.cpu_count=max_nodes
		n=self.cpu_count
		
		self.fpath = os.path.join(tempfile.gettempdir(),'mp')
		os.makedirs(self.fpath, exist_ok=True)
		os.makedirs(self.fpath+'/slaves', exist_ok=True)
		
		self.slaves=[slave() for i in range(n)]
		pids=[]
		for i in range(n):
			self.slaves[i].confirm(i,initcommand,fpath) 
			pid=str(self.slaves[i].p_id)
			if int(i/5.0)==i/5.0:
				pid='\n'+pid
			pids.append(pid)
		self.send_holdbacks(holdbacks)
		pstr="""Multi core processing enabled using %s cores. \n
Master PID: %s \n
Slave PIDs: %s"""  %(n,os.getpid(),', '.join(pids))
		print (pstr)

	def send_dict(self, d):
		for i in cpu_ids:
			self.slaves[i].send('dict',d)
		a=0
		
			
	def quit(self):
		for i in self.slaves:
			i.p.stdout.close()
			i.p.stderr.close()
			i.p.stdin.close()
			i.p.kill()
			i.p.wait()
			
class Tasks:
	def __init__(self, mp, tasks, progress_bar=None):
		"""tasks is a list of string expressions to be executed. All variables in expressions are stored in the dictionary sent to the slaves
		if remote=True, the list must be a list of tuples, where the first item is the expression and the second is the variable that should be returned\n
		If wait_and_collect=True, wait_and_collect MUST be called at a later stage for each node. If not the nodes will be lost"""
		
		mp.send_dict_by_file_receive()
		self.mp = mp
		self.n=len(tasks)
		self.tasks = tasks
		self.sent=min((mp.cpu_count, self.n))
		self.d_arr=[]
		self.msg = 'exec'
		self.progress_bar = progress_bar
		self.q = Queue()


		for i in range(self.sent):
			mp.slaves[i].send(self.msg, self.tasks[i])#initiating the self.cpus first evaluations
			t=Thread(target=mp.slaves[i].receive,args=(self.q,), daemon=True)
			t.start()

	def collect(self):
		"""Waiting and collecting the sent tasks. """

		if len(self.tasks) == 0:
			return		

		for i in range(self.n):
			d,s = self.q.get()
			self.d_arr.append([d,s])			
			if self.sent<len(self.tasks):
				self.mp.slaves[s].send(self.msg, self.tasks[self.sent])#supplying additional tasks for returned cpus
				t=Thread(target=self.mp.slaves[s].receive,args = (self.q,),daemon=True)
				t.start()
				self.sent += 1
			
			if not self.progress_bar is None:
				pb_func,pb_min,pb_max,text = self.progress_bar			
				pb_func(pb_min+(pb_max-pb_min)*i/self.n,text)
		d,d_node=get_slave_dicts(self.d_arr)
		
		return d,d_node
	
	
def get_slave_dicts(d_arr):

	d_var={}
	d_node={}
	for d,s in d_arr:
		for key in d:
			if key in d_var:
				raise RuntimeWarning('Slaves returned identical variable names. Some variables will be overwritten')
			d_var[key]=d[key]
			d_node[key]=s
	return d_var,d_node



class slave():
	"""Creates a slave"""
	command = [sys.executable, "-u", "-m", "multi_core_slave.py"]


	def __init__(self):
		"""Starts local worker"""
		cwdr=os.getcwd()
		os.chdir(os.path.dirname(__file__))
		self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		os.chdir(cwdr)
		self.t=transact(self.p.stdout,self.p.stdin)
		
	def confirm(self,slave_id,initcommand,fpath):
		self.p_id = self.receive()
		self.slave_id=slave_id
		self.send('init_transact',(initcommand,slave_id,fpath))
		self.initcommand=initcommand
		self.fpath=fpath
		pass

	def send(self,msg,obj):
		"""Sends msg and obj to the slave"""
		if not self.p.poll() is None:
			raise RuntimeError('process has ended')
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
		self.r = read
		self.w = write

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

	def execute(self,expr,progress_bar=None, collect=True):
		"""For submitting multiple functions to be evalated. expr is an array of strings to be evaluated"""
		"""If collect == True, collect results immediately, else results are collected later with collect()"""
		
		self.progress_bar = progress_bar
		self.tasks = Tasks(self.master, expr,progress_bar=progress_bar)
		if not collect:
			return
		d,d_node = self.tasks.collect()
		self.tasks = None
		for i in d:
			self.d[i]=d[i]
		return self.d
	
	def collect(self):
		if self.tasks==None:
			raise RuntimeError('Tasks has all ready been collected')
		d,d_node = self.tasks.collect()
		self.tasks = None
		for i in d:
			self.d[i]=d[i]		
		return self.d
	
	def send_dict(self,d,cpu_ids=None,command=''):
		for i in d:
			self.d[i]=d[i]		
		self.master.send_dict(d,cpu_ids,command)
			
	def quit(self):
		self.master.quit()
		
	def listen(self, tasks, outbox):
		self.listen_task = Listen(self.master, tasks, outbox)
		return self.listen_task


def obtain_fname(name):

	path=os.path.abspath(name)
	path_dir=os.path.dirname(path)
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)	

	return path