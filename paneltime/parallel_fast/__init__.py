#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import pickle
from queue import Queue
from threading import Thread
import tempfile
from . import transact


class Master():
	"""A class that handles multi processing"""
	def __init__(self, n, import_dir):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """

		self.fpath = os.path.join(tempfile.gettempdir(),'mp')
		os.makedirs(self.fpath, exist_ok=True)
		self.import_dir = import_dir
		self.n_slaves = n
		self.slaves=[slave() for i in range(n)]
		self.q = Queue()
		pids=[]
		for i in range(n):
			self.slaves[i].confirm(i,self.fpath) 
			pid=str(self.slaves[i].p_id)
			if int(i/5.0)==i/5.0:
				pid='\n'+pid
			pids.append(pid)

		pstr="""Multi core processing enabled using %s cores. \n
Master PID: %s \n
Slave PIDs: %s"""  %(n,os.getpid(),', '.join(pids))
		print (pstr)

	def send_dict(self, d):
		f = tempfile.NamedTemporaryFile()
		fname = f.name
		f.close()
		f = open(fname,'wb')
		pickle.dump(d,f)
		f.close()
		for s in self.slaves:
			s.send('dict',fname)
			t=Thread(target=s.receive,args=(self.q,), daemon=True)
			t.start()
		self.collect()
		#threading.Thread(target=delayed_close, args=(fname,)) 

		a=0
		
			
	def quit(self):
		for i in self.slaves:
			i.p.stdout.close()
			i.p.stderr.close()
			i.p.stdin.close()
			i.p.kill()
			i.p.wait()
			

	def run(self, tasks, operation):
		"""tasks is a list of string expressions to be executed. All variables in expressions are stored 
		in the dictionary sent to the slaves"""
		if not self.q.empty():
			self.collect()
		if type(tasks) == str:
			tasks = [tasks]*self.n_slaves
		for i, s in enumerate(self.slaves):
			s.send(operation, tasks[i])#initiating the self.cpus first evaluations
			t=Thread(target=s.receive,args=(self.q,), daemon=True)
			t.start()

	def exec(self, task):
		self.run(task, 'exec')

	def eval(self, task):
		self.run(task, 'eval')

	def collect(self):
		"""Waiting and collecting the sent tasks. """
		d = {}
		while not self.q.empty():	
			ds,s = self.q.get()
			d[s] = ds			
		return d
	
	def import_module(self, name):
		if not name[-3:] == '.py':
			name+= '.py'
		tasks = [os.path.join(self.import_dir, name)]*self.n_slaves
		self.run(tasks, 'import')
	


class slave():
	"""Creates a slave"""
	command = [sys.executable, "-u", "-m", "slave.py"]


	def __init__(self):
		"""Starts local worker"""
		cwdr=os.getcwd()
		os.chdir(os.path.dirname(__file__))

		self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		os.chdir(cwdr)
		self.t=transact.Transact(self.p.stdout,self.p.stdin)
		
	def confirm(self,slave_id,fpath):
		self.p_id = self.receive()
		self.slave_id=slave_id
		self.send('init_transact',
							(slave_id, os.path.join(fpath, f''), )
							)
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




import threading
import time

def delayed_close(fname):
	i=0
	for i in range(200):
		i+=1
		try:
			time.sleep(1)  # Thread sleeps here, minimal resource usage
			os.remove(fname)        # Close the file after the delay
			print(f"{fname} closed at iteration {i}")
		except:
			pass
		
