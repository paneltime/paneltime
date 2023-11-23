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
	def __init__(self, n):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """

		self.fpath = 'mp'#os.path.join(tempfile.gettempdir(),'mp')
		os.makedirs(self.fpath, exist_ok=True)
		os.makedirs(self.fpath+'/slaves', exist_ok=True)
		
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
		os.remove(fname)
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

		for i, s in enumerate(self.slaves):
			s.send(operation, tasks[i])#initiating the self.cpus first evaluations
			t=Thread(target=s.receive,args=(self.q,), daemon=True)
			t.start()

	def exec(self, task):
		self.run(task, )

	def eval(self, task):
		self.run(task, 'eval')

	def collect(self):
		"""Waiting and collecting the sent tasks. """
		d = {}
		while not self.q.empty():	
			ds,s = self.q.get()
			d[s] = ds			
		return d
	


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
		self.send('init_transact',(slave_id,fpath))
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




