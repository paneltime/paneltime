#!/usr/bin/env python
# -*- coding: utf-8 -*-


#THIS DOES NOT WORK!!!
import subprocess
import os
import pickle

def main():
	a=browser()
	a.send("Hello World")
	d=0

class browser:
	"""Creates a slave"""
	command = ["start",os.path.join(os.path.dirname(__file__),"io.html")]

	def __init__(self):
		"""Starts browser"""
		command='start '+os.path.join(os.path.dirname(__file__),"io.html")
		self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
	
		self.t=transact(self.p.stdout,self.p.stdin)


	def send(self,msg):
		"Sends msg to the browser"
		self.t.send((msg,))          

	def receive(self):
		"waits to recive message from the borowser"
		answ=self.t.receive()
		return answ

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
			
			
main()