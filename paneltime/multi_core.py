#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import pickle
import datetime
from queue import Queue
from threading import Thread
import time
import tempfile
import random
import string


class Master():
	"""creates the slaves"""
	def __init__(self, max_nodes, fpath):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """
		self.pending_cpus=[]
		if max_nodes is None:
			self.cpu_count=os.cpu_count()
		else:
			self.cpu_count=max_nodes
		n=self.cpu_count
		self.dict_file = create_temp_file(self.cpu_count)
		self.fpath = makepath(fpath)
		self.slaves=[Slave() for i in range(n)]
		pids=[]
		for i in range(n):
			self.slaves[i].confirm(i, self.fpath, self.dict_file) 
			pid=str(self.slaves[i].p_id)
			if int(i/5.0)==i/5.0:
				pid='\n'+pid
			pids.append(pid)
		pstr="""Multi core processing enabled using %s cores. \n
Master PID: %s \n
Slave PIDs: %s"""  %(n,os.getpid(),', '.join(pids))
		print (pstr)

	def send_dict(self, d):
		f = open(self.dict_file, 'wb')
		pickle.dump(d, f)  
		f.close()
		r = []
		for i in range(self.cpu_count):
			self.slaves[i].send('transfer dictionary',None)
			r.append(self.slaves[i].receive())
			
	def exec(self, tasks, name):
		#todo: add capebilities for useing more nodes than cpu_count
		r = []
		if tasks is None:
			tasks = [None]*self.cpu_count
		if type(tasks)==str:
			tasks = [tasks]*self.cpu_count
		for i in range(min((len(tasks), self.cpu_count))):
			self.slaves[i].send(name,tasks[i])
		for i in range(min((len(tasks), self.cpu_count))):	
			r.append(self.slaves[i].receive())
		return r
			
			
	def callback(self, proc_name, outbox = {}, s_id = None):
		d=[]
		if not s_id is None:
			self.slaves[s_id].send('callback',(proc_name,outbox))
			return self.slaves[s_id].receive()
		for i in range(self.cpu_count):
			self.slaves[i].send('callback',(proc_name, outbox))
			d.append(self.slaves[i].receive())
		return d
	
	def wait_untill_done(self):
		while self.any_alive():
			time.sleep(0.01)
		
	def any_alive(self):
		return sum(self.exec(None, 'check_state'))

			
	def quit(self):
		for i in self.slaves:
			i.p.stdout.close()
			i.p.stderr.close()
			i.p.stdin.close()
			i.p.kill()
			i.p.cleanup()
			
def makepath(fpath):
	fpath=os.path.join(fpath,'mp')
	os.makedirs(fpath, exist_ok=True)
	os.makedirs(fpath+'/slaves', exist_ok=True)	
	return fpath

	
class Slave():
	"""Creates a slave"""
	command = [sys.executable, "-u", "-m", "multi_core_slave.py"]


	def __init__(self):
		"""Starts local worker"""
		cwdr=os.getcwd()
		os.chdir(os.path.dirname(__file__))
		self.p = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		os.chdir(cwdr)
		self.t=transact(self.p.stdout,self.p.stdin)
		
	def confirm(self,slave_id, path, f_dict):
		self.p_id = self.receive()
		self.slave_id=slave_id
		self.send('init_transact', (slave_id, path, f_dict))
		pass

	def send(self,msg, obj):
		"""Sends msg and obj to the slave"""
		if not self.p.poll() is None:
			raise RuntimeError('process has ended')
		self.t.send((msg, obj))     

	def receive(self):
		answ=self.t.receive()
		return answ

	
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

	def receive(self):
		r=getattr(self.r,'buffer',self.r)
		u= pickle.Unpickler(r)
		try:
			return u.load()
		except pickle.UnpicklingError as e:
			r.seek(0)
			u= pickle.Unpickler(r)
			return u.load()
		except EOFError as e:
			if e.args[0]=='Ran out of input':
				raise RuntimeError("""An error occured in one of the spawned sub-processes. 
Check the output in "slave_errors.txt' in your working directory or 
run without multiprocessing\n %s""" %(datetime.datetime.now()))
			else:
				raise RuntimeError('EOFError:'+e.args[0])

def write(f,txt):
	f = open(f, 'w')
	f.write(txt)
	f.flush()
	f.close()
	
def read(file):
	f = open(file, 'r')
	s = f.read()
	f.close()
	return s

def create_temp_file(cpu_count):
	tdir = tempfile.gettempdir()
	f = os.path.join(tdir,gen_file_name(True))
	try:#deleting old file
		fold = read(f)
		os.remove(fold)
	except Exception as e:
		print(e)
		a=0
	
	f_dict = gen_file_name()
	f_ = open(f_dict,'w')
	f_.close()
	write(f, f_dict)
	return f_dict
	
def gen_file_name(seed=False):
	if seed:
		f = os.path.join('\\'.join(__file__.split('\\')[:-2]),os.path.join('.git','config'))
		seed = os.path.getmtime(f)
		random.seed(seed)
	else:
		random.seed()
	tdir = tempfile.gettempdir()
	fname = ''.join(random.choice(string.ascii_uppercase+string.ascii_lowercase + string.digits) for _ in range(20))
	fname = os.path.join(tdir,fname)
	return fname
		
		