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
import parallel_slave
import hashlib
import psutil
import signal

class Parallel():
	"""creates the slaves"""
	def __init__(self, max_nodes, fpath, run_parallel = True, callback_active = True):
		"""module is a string with the name of the modulel where the
		functions you are going to run are """
		if max_nodes is None:
			self.cpu_count=os.cpu_count()
		else:
			self.cpu_count=max_nodes
		self.callback_active = callback_active
		n=self.cpu_count
		self.is_parallel = run_parallel
		self.slave_path = os.path.join(os.path.dirname(__file__), "parallel_node.py")
		self.kill_orphans(fpath)
		self.dict_file = create_temp_files(self.cpu_count, fpath)
		self.fpath = makepath(fpath)
		self.slaves=[Slave(run_parallel, n, self.slave_path) for i in range(n)]
		self.final_results = {}
		self.n_tasks = {}
		pids=[]
		for i in range(n):
			self.slaves[i].confirm(i, self.fpath) 
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
		return self.async_send_receive('transfer dictionary', self.dict_file)

			
	def exec(self, tasks, name):
		#Handles at most as many tasks as cpus for the moment
		if tasks is None:
			tasks = [None]*self.cpu_count
		if type(tasks)==str:
			tasks = [tasks]*self.cpu_count
		self.n_tasks[name] = len(tasks)
		self.final_results[name] = [None]*self.n_tasks[name]
		return self.async_send_receive(name, tasks)
			
	def callback(self, name, outbox = {}, s_id = None, collect_finnished_procs = True):
		if not self.callback_active:
			return [{}]*self.cpu_count
		d=[]
		if not s_id is None:
			self.slaves[s_id].send('callback',(name,outbox))
			return self.slaves[s_id].receive()
		response = self.async_send_receive('callback', (name, outbox))
		if not collect_finnished_procs:
			return response
		is_running = self.check_state(name)
		for i in range(self.n_tasks[name]):
			if (not is_running[i]) and (self.final_results[name][i] is None):
				r = self.collect(name, i)
				self.final_results[name][i] = r
				if r is None:
					self.final_results[name][i] = 'No result'
			if not self.final_results[name][i] is None:
				response[i] = self.final_results[name][i]
		return response

	
	def collect(self, name, sid=None):
		if not sid is None:
			self.slaves[sid].send('collect', name)
			msg = self.slaves[sid].receive()	
			return 	msg

		return self.async_send_receive('collect', name)
		
	def async_send_receive(self,msg, obj):
		r = []
		q = Queue()
		if not type(obj) == list:
			obj = [obj]*self.cpu_count
		for i in range(len(obj)):
			self.slaves[i].send(msg, obj[i])
			t = Thread(target = self.slaves[i].receive, args = (q, ), daemon= True)
			t.start()
		r = [None]*len(obj)
		for i in range(len(obj)):
			res, sid = q.get()
			r[sid] = res
		return r
		
	def count_alive(self):
		return sum(self.exec(None, 'check_state'))
	
	def check_state(self, name = None):
		return self.exec(name, 'check_state')


			
	def quit(self):
		for i in self.slaves:
			i.p.stdout.close()
			i.p.stderr.close()
			i.p.stdin.close()
			i.p.kill()
			i.p.cleanup()
			
	def kill_orphans(self, fpath):
		for proc in psutil.process_iter():
			if proc.name() == 'python.exe':
				if proc.cmdline()[-1]==self.slave_path:
					pid = proc.pid
					os.kill(pid, signal.SIGTERM)
					print(f"killed pid {pid}")					

				
			
			
def makepath(fpath):
	fpath=os.path.join(fpath,'mp')
	os.makedirs(fpath, exist_ok=True)
	os.makedirs(fpath+'/slaves', exist_ok=True)	
	return fpath

	
class Slave():
	"""Creates a slave"""
	def __init__(self, run_parallel, n, path):
		"""Starts local worker"""
		self.n_nodes = n
		if run_parallel:
			command = f'"{sys.executable}" -u "{path}"'	
			self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			sys.stderr = self.p.stderr
			self.t = Transact(self.p.stdout,self.p.stdin)
			
		else:
			qin, qout = Queue(), Queue()
			self.p = ThreadProcess(parallel_slave.run, (TransactThread(qout, qin),False))
			self.p.start()
			self.t = TransactThread(qin, qout)

		
	def confirm(self,slave_id, path):
		self.t.set(f'master {self.n_nodes}', self.n_nodes, path, slave_id)
		self.slave_id=slave_id
		self.p_id = self.receive()
		self.send('init_transact', (slave_id, path, self.n_nodes))
		pass

	def send(self,msg, obj):
		"""Sends msg and obj to the slave"""
		if not self.p.poll() is None:
			raise RuntimeError('process has ended')
		self.t.send((msg, obj))     

	def receive(self, q = None):
		answ=self.t.receive()
		if q is None:
			return answ
		q.put((answ,self.slave_id))

	
	def kill(self):
		self.p.kill()

class Transact():
	"""Local worker class"""
	def __init__(self,read, write):
		self.r = read
		self.w = write
		self.name = None
		self.slave_id = None
		self.time = time.time()
		self.f = None

	def set(self,name, n_nodes, path, slave_id):
		self.name = name
		self.n_nodes = n_nodes
		self.path = path
		self.slave_id = slave_id
		path = os.path.join(path,'debug')
		if not os.path.exists(path):
			os.mkdir(path)
		path = os.path.join(path, f'{self.name}_{self.slave_id}_{n_nodes}.txt')
		self.f = open(path, 'w',1)		

	def send(self,msg):
		self.debug_output('send',msg)
		w=getattr(self.w,'buffer',self.w)
		pickle.dump(msg,w)
		w.flush()

	def receive(self):
		r=getattr(self.r,'buffer',self.r)
		u= pickle.Unpickler(r)
		try:
			msg = u.load()
			self.debug_output('recieve', msg)
			return msg
		except pickle.UnpicklingError as e:
			r.seek(0)
			u= pickle.Unpickler(r)
			return u.load()
		except EOFError as e:
			if e.args[0]=='Ran out of input':
				raise RuntimeError(
					f"An error in {self.name} occured in sub-process {self.slave_id}." 
					f"Check the output in 'slave_errors.txt' in your working directory or "
					f"run without multiprocessing\n {datetime.datetime.now()}"
				)
			else:
				raise RuntimeError('EOFError:'+e.args[0])
			
	def debug_output(self, direction,msg):
		return
		if self.f is None:
			return
		self.f.write(f'{direction}: {self.name}, {self.slave_id}\n{time.time()-self.time}:\n{str(msg)[:30]}\ntime:{time.time()}\n')	
		self.time = time.time()		
		
			
class ThreadProcess(Thread):
	"""Starts local worker"""
	def __init__(self, target, args):
		super().__init__(target=target, args = args)
	def kill(self):
		raise RuntimeError("No kill procedure written yet")
	def poll(self):
		if self.is_alive():
			return None
		else:
			return 1
			
class TransactThread(Transact):
	"""Thread version of Transact"""
	def __init__(self,read, write):
		self.r = read
		self.w = write
		self.name = None
		self.slave_id = None
		self.time = time.time()
		self.f = None		
		
	def set(self,name, n_nodes, path, slave_id):
		self.name = name
		self.n_nodes = n_nodes
		self.path = path
		self.slave_id = slave_id
		path = os.path.join(path,'debug')
		if not os.path.exists(path):
			os.mkdir(path)
		path = os.path.join(path, f'{self.name}_{self.slave_id}_{n_nodes}.txt')
		self.f = open(path, 'w',1)	

	def send(self,msg):
		self.debug_output('send',msg)
		self.w.put(msg) 

	def receive(self):
		msg = self.r.get()
		self.debug_output('recieve',msg)
		return msg
	
	def debug_output(self, direction,msg):
		return
		if self.f is None:
			return
		self.f.write(f'{direction}: {self.name}, {self.slave_id}\n{time.time()-self.time}:\n{str(msg)[:30]}\ntime:{time.time()}\n')	
		self.time = time.time()			
			

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

def create_temp_files(cpu_count, fpath):
	number_of_files = 1

	try:#deleting old file
		f = get_file(fpath)
		folds = eval(read(f))
		for i in folds:
			os.remove(i)
	except Exception as e:
		print(e)
		a=0
	
	ftmp = []
	for i in range(number_of_files):
		f_name = gen_file_name()
		f_ = open(f_name,'w')
		f_.close()
		ftmp.append(f_name)
	write(f, str(ftmp))
	
	if number_of_files==1:
		ftmp = ftmp[0]
	return ftmp

def get_file(fpath):
	tdir = tempfile.gettempdir()
	f = os.path.join(tdir,gen_file_name(fpath))
	return f
	
def gen_file_name(seed=False):
	if not seed==False:
		f = os.path.join(os.sep.join(__file__.split(os.sep)[:-2]),os.path.join('.git','config'))
		n = len(seed)
		seed = int(hashlib.sha1(seed.encode('utf-8')).hexdigest(), 16)
		seed = os.path.getmtime(f) + seed
		random.seed(seed)
	else:
		random.seed()
	tdir = tempfile.gettempdir()
	fname = ''.join(random.choice(string.ascii_uppercase+string.ascii_lowercase + string.digits) for _ in range(20))
	fname = os.path.join(tdir,fname)
	return fname
		
		

		