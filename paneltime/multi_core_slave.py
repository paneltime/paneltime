
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import multi_core
import traceback
import datetime
import pickle
import gc
import inspect
import time
from threading import Thread
import subprocess
import callback


class Slave:
	def __init__(self, t, s_id, path, f_dict):
		self.t = t
		self.id = s_id
		self.f_dict = f_dict
		fstdout = os.path.join(path,f'slaves/{s_id}.txt')
		sys.stdout = open(fstdout,'w',1)
		self.d = dict()

		self.processes = Processes(f_dict)
		
		while 1:
			self.wait_for_orders()

		
	def wait_for_orders(self):
	
		(msg,obj) = self.t.receive()
		response = None
		if msg == 'kill':
			sys.exit()
			response = True
			
		elif msg == 'callback':
			name, d = obj
			if name in self.processes:
				self.processes[name].callback.callin(**d)
				if len(d):
					print(f"got dict:{d}")
				response = self.processes[name].callback.outbox
			else:
				response = {'empty':True}
		elif msg == 'check_state':
			response = self.processes.anyalive()
		else:
			self.processes[msg] = obj
		t.send(response)		
		gc.collect()
	


class Processes(dict):
	
	def __init__(self,f_dict):
		self.d = {}
		self.f_dict = f_dict
		self.alive = {}
		
	def __setitem__(self, name, task):
		super().__setitem__(name, Process(name, task, self))
		
	def set_status(self,name, state):
		self.alive[name] = state
		
	def anyalive(self):
		return sum([self.alive[k] for k in self.alive])>0
		
		
		
class Process:
	def __init__(self, name, task, parent):			
		self.name = name
		self.parent = parent
		self.task = task
		self.d = parent.d
		self.d['callback'] = callback.CallBack()
		self.callback = self.d['callback']	
		
		if name == 'transfer dictionary':
			thread = Thread(target=self.get_dict)
		else:
			thread = Thread(target=self.run)

		thread.start()		
		
	def run(self):
		try:
			self.parent.set_status(self.name, True)
			exec(self.task, globals(), self.d)
			self.parent.set_status(self.name, False)
		except Exception as e:
			traceback.print_exc(file = sys.stdout)
			print(f"task: {self.task}")
			raise RuntimeError(e)
			
		
	def get_dict(self):		
		try:
			self.parent.set_status(self.name, True)
			f = open(self.parent.f_dict,'rb')
			d = None
			d =  pickle.load(f)
			f.close()
			d =  dict(d)	
			for k in d:
				self.d[k] = d[k]
			self.parent.set_status(self.name, False)	
		except Exception as e:
			traceback.print_exc(file = sys.stdout)
			print(f"task: {d}")
			raise RuntimeError(e)

		
def write(f,txt):
	f = open(f, 'w', 1)
	f.write(str(txt))
	f.flush()
	f.close()


try: 
	t = multi_core.transact(sys.__stdin__, sys.__stdout__)
	#Handshake:
	t.send(os.getpid())
	path='.'
	msg, (s_id, path, f_dict)=t.receive()
	#error handling
	fname = os.path.join(path,'slave_errors.txt')
	#Wait for instructions:
	Slave(t, s_id, path, f_dict)
	
except Exception as e:
	f = open(fname,'w', 1)
	f.write('SID: %s      TIME:%s \n' %(s_id,datetime.datetime.now()))
	traceback.print_exc(file = f)
	f.flush()
	f.close()
	raise RuntimeError(e)