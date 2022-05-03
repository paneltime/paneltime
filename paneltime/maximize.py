#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import maximize_num
import loglikelihood as lgl
import pickle


class LLClass:
	def __init__(self, panel, constr):
		self.panel = panel
		self.constr = constr
		self.ll = None
	
	def LL(self,x):
		self.ll = lgl.LL(x, self.panel, self.constr)
		return self.ll.LL


def lnsrch(ll, direction, panel):
	
	p = direction.dx
	g = direction.g
	stpmax=100*max((abs(sum(ll.args.args_v**2)))**0.5,float(len(ll.args.args_v))) 
	ll_class = LLClass(panel, direction.constr)
	fret,x,check, _ , lmbda, rev = maximize_num.lnsrch(ll.args.args_v ,ll.LL ,g ,p ,stpmax ,ll_class.LL ) 
	ll=ll_class.ll
	msg=''
	if rev:
		msg='direction reversed'
	return ll,msg,lmbda,True,rev	
	
def maximize(panel,direction,mp,args,channel,msg_main="",log=[]):
	"""Maxmizes logl.LL"""

	its,incr, msg,lmbda,rev =0, 0, '',1,False
	ll=direction.init_ll(args)
	
	if mp is None:
		fret,args,hessin,its,Convergence = maximize_num.maximize(panel,args.args_v,2,0.001)
	else:
		comm = Comm(mp, panel, args, ll, direction)

	
	#maximize_num.maximize_test(panel,args.args_v, 8,0)
	
	po=printout(channel,panel,ll,direction,msg_main)
	if not printout_func(0.0,'Determining direction',ll,its,direction,incr,po,0):return ll,direction,po
	#direction.start_time=time.time()
	while 1:
		direction.calculate(ll,its,msg,incr,lmbda,rev)
		direction.set(ll.args)
		LL0=ll.LL
		#Convergence test:
		conv=convergence_test(direction, its, ll,panel,incr)
		if conv: 
			printout_func(1.0,"Convergence on zero gradient; maximum identified",ll,its,direction,incr,po,3,'done')
			comm.terminate()
			return ll,direction,po
		if lmbda==0:
			printout_func(1.0,"No increase in linesearch without convergence. This should not happen. Contact Espen.Sirnes@uit.no",ll,its,direction,incr,po,1,"err")
		#if not printout_func(1.0,"",ll,its,direction,incr,po,1,task='linesearch'):return ll,direction,po
		
		ll,msg,lmbda,ok,rev=lnsrch(ll,direction, panel)
		incr=ll.LL-LL0
		ll, H = comm.communicate(ll, direction.H)
		
		#if not printout_func(1.0,msg,ll,its,direction,incr,po,0):return ll,direction,po
		print(f"{ll.LL}:{its}")

		its+=1

class Comm:
	def __init__(self, mp, panel, args, ll, direction):
		self.mp = mp
		self.panel = panel
		self.direction = direction
		f = panel.input.tempfile.tempfile(delete=False)
		self.f_write = f.name.replace('\\','/')
		f.close()
		self.write('', ll.LL, args.args_v, direction.H)
		self.spawn()
		self.t = time.time()
		
	def write(self, command, LL=None, args=None, H=None):
		f = open(self.f_write, 'wb')
		pickle.dump((command, LL, args, H), f)
		f.flush()
		f.close()
		
	def read(self, fname):
		try:
			f = open(fname, 'rb')
			response, LL, args, hessin = pickle.load(f)
		except EOFError:
			f.close()
			return False, None, None, None, None
		f.close()
		return True, response, LL, args, hessin
		
		
	def spawn(self):
		tasks=[]
		lamdas=[1,4,8]
		self.f_read_files=[]
		for i in range(len(lamdas)):
			f = self.panel.input.tempfile.tempfile(delete=False)
			fname=f.name.replace('\\','/')
			f.close()
			self.f_read_files.append(fname)
			tasks.append(f"maximize_num.maximize(panel, f_write='{fname}',f_read='{self.f_write}', id)")
		self.mp.execute(tasks, wait_and_collect=False)
		
	def terminate(self):
		if self.mp is None:
			return
		self.write('abort')
		self.mp.wait_and_collect()

	def communicate(self, ll, H):
		if time.time() - self.t<1 or self.mp is None:		
			return ll, H
		LLs = [ll.LL]
		xs = [ll.args.args_v]
		h = []
		rf=list(self.f_read_files)
		while len(rf):
			f = rf.pop(0)
			OK, response, fret, x, hessin = self.read(f)
			if OK:
				LLs.append(fret)
				xs.append(x) 
				h.append(hessinv(hessin, H))				
			else:
				rf.append(f)
		i=np.argmax(LLs)
		if LLs[i]>ll.LL:
			ll=lgl.LL(xs[i], self.panel, self.direction.constr)
		self.write('', ll.LL, ll.args.args_v, self.direction.H)
		self.t = time.time()
		return ll, H
	
	def respawn()
	
def hessinv(hessin, H):
	try:
		hess = np.inv(hessin)
	except:
		return H
	return hess

def potential_gain(direction):
	"""Returns the potential gain of including each variables, given that all other variables are included and that the 
	quadratic model is correct. An alternative convercence criteria"""
	dx=direction.dx
	n=len(dx)
	g=direction.g
	rng=np.arange(len(dx))
	dxLL=dx*0
	dxLL_full=(sum(g*dx)+0.5*np.dot(dx.reshape((1,n)),np.dot(direction.H,dx.reshape((n,1)))))[0,0]
	for i in range(len(dx)):
		dxi=dx*(rng!=i)
		dxLL[i]=dxLL_full-(sum(g*dxi)+0.5*np.dot(dxi.reshape((1,n)),np.dot(direction.H,dxi.reshape((n,1)))))[0,0]
	return dxLL	
	
	
def convergence_test(direction,its,ll, panel, incr, min_iter=4):
	pgain=potential_gain(direction)
	#print(pgain)
	
	if incr>0:
		args_archive = panel.input.args_archive
		args_archive.save(ll.args,True)

	conv=(max(pgain)<panel.options.tolerance.value) and (max(np.abs(direction.dx_norm))<0.0001)
	conv*=(its>=min_iter)
	return conv

		
def printout_func(percent,msg,ll,its,direction,incr,po,update_type, task=""):
	po.printout(ll,its+1,direction,incr,update_type)	
	return direction.progress_bar(percent,msg,task=task)
		
class printout:
	def __init__(self,channel,panel,ll,direction,msg_main,_print=True,):
		self._print=_print
		self.channel=channel
		if channel is None:
			return
		channel.set_output_obj(ll, direction,msg_main)

		
		
	def printout(self,ll, its, direction,incr,update_type):
		if self.channel is None and self._print:
			print(ll.LL)
			return
		if not self._print:
			return
		if update_type>0:
			self.channel.update_after_direction(direction,its)
		elif update_type==0 or update_type==2:
			self.channel.update_after_linesearch(direction,ll,incr)

