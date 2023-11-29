
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import datetime


class Transact():
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
Check the output in the folder "mp' in your temp directory or 
run without multiprocessing\n %s""" %(datetime.datetime.now()))
			else:
				raise RuntimeError('EOFError:'+e.args[0])

def write(f,txt):
	f.write(str(txt)+'\n')
	f.flush()