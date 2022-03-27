#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import loglikelihood as logl
import output
import sys
from tkinter import _tkinter
import time
import loglikelihood as lgl
import direction as drctn
import maximize_num
import maximize_num2




def lnsrch(ll, direction,mp,its,incr,po,prev_dx,dx,max_its=12,convex_action='reverse',single_core=False):
	panel=direction.panel
	rev=False
	
	args=ll.args.args_v
	g = direction.g
	
	LL0=ll.LL
	constr=direction.constr
	
	
	if np.sum(g*dx)<0:
		if convex_action=='reverse':
			dx=-dx
			direction.progress_bar(text="Convex function at evaluation point, direction reversed")
			rev=True
		elif convex_action=='abort':
			return ll,'Convex function: aborted linesearch',0,False,False

	dx_len=sum(dx**2)**0.5
	for i in range(3):
		LL0=ll.LL
		ll,msg,lmbda,ok, res=lnsrch_master(ll.args.args_v, dx,panel,constr,mp,LL0,ll,max_its,single_core)
		break
		if i>0:
			print(f"#{i} LL: {ll.LL}  DLL: {ll.LL-LL0}   lmda: {lmbda}")
		if lmbda<0.1 and False:
			break
		if i<2:
			direction.calc_gradient(ll)
			g=direction.g
			dx = g*dx_len/sum(g**2)**0.5	
		

	if  ll.LL/panel.NT<-1e+15:
		msg='The maximization agorithm has gone mad. Resetting the argument to initial values'
		ll=logl.LL(panel.args.args_restricted,panel)
		return ll,msg,0,False
	
	return ll,msg,lmbda,ok,rev 
		
		
def solve_square_func(f0,l0,f05,l05,f1,l1,default=None):
	try:
		b=-f0*(l05+l1)/((l0-l05)*(l0-l1))
		b-=f05*(l0+l1)/((l05-l0)*(l05-l1))
		b-=f1*(l05+l0)/((l1-l05)*(l1-l0))
		c=((f0-f1)/(l0-l1)) + ((f05-f1)/(l1-l05))
		c=c/(l0-l05)
		if c<0 and b>0:#concave increasing function
			return -b/(2*c)
		else:
			return default
	except:
		return default
	
	
def lnsrch_master(args, dx,panel,constr,mp,LL0,ll,max_its,single_core):
	start=0
	end=2.0
	msg=''
	#single_core is for debug. It has been tested, and even with a relatively small sample, multicore is much faster.
	for i in range(max_its):
		delta=(end-start)/(mp.master.cpu_count-1-(i==0))
		res=get_likelihoods(args, dx, panel, mp,delta,start,single_core, constr)
		if i==0:
			res0=res[0]		
		if (res[0,1]==0 and i==0) or (res[0,0]<=res0[0] and i>0) or np.isnan(res[0,0]):#best was no change
			start=delta/mp.master.cpu_count
			end=delta
		else:
			break
	if i>0:
		if np.max(res[:,0])==res0[0]:
			return ll,'No increase in linesearch',0,False
		res=np.append([res0],res,0)#ensuring the original is in the LL set
		srt=np.argsort(res[:,0])[::-1]
		res=res[srt]
	res=remove_nan(res)
	if LL0>res[0,0]:
		s='Best result in linesearch is poorer than the starting point. You may have discovered a bug, please notify espen.sirnes@uit.no'
		print(s)
		return ll,s,res[0,0]-LL0,False, res
	try:
		lmda=solve_square_func(res[0,0], res[0,2],res[1,0], res[1,2],res[2,0], res[2,2],res[0,2])
		1/lmda#Throws an exception
		ll2=logl.LL(args+lmda*dx,panel,constr)
		if ll2.LL<res[0,0]:
			raise RuntimeError('Something wrong with ll. You may have discovered a bug, please notify espen.sirnes@uit.no')
	except:
		ll2, lmda = mp.remote_recieve(f'f{res[0,1]}',res[0,1],single_core), res[0,2]
	if ll2.LL is None or lmda==0:
		return ll,'No increase in linesearch',0,False, res
	msg=f"Linesearch success ({round(lmda,6)} of Newton step). "
	return ll2,msg,lmda,True, res

def remove_nan(res):
	r=[]
	for i in res:
		if not np.isnan(i[0]):
			r.append(i)
	return np.array(r)	
	
				
def get_likelihoods(args, dx,panel,mp,delta,start,single_core, constr):
	lamdas=[]
	a=[]
	for i in range(mp.master.cpu_count):
		lmda=start+i*delta
		a.append(list(args+lmda*dx))
		lamdas.append(lmda)
	res=likelihood_spawn(a,lamdas,single_core,mp,{'lgl':lgl, 'panel':panel, 'constr':constr})
	return res

	
def likelihood_spawn(args, return_info,single_core,mp,d):
	"Returns a list of a list with the LL of the args and return_info, sorted on LL"
	expr=[]	
	n=len(args)
	for i in range(n):
		expr.append([
			"try:\n"
			f"	f{i}=lgl.LL({list(args[i])}, panel,constr)\n"
			f"	LL{i}=f{i}.LL\n"
			"except:\n"
			f"	LL{i}=None\n"
			
			, f'LL{i}'])
		a=0
	d,d_node=mp.remote_execute(expr,single_core,d=d)
	res=[]
	for i in range(n):
		key=f'LL{i}'
		if not d[key] is None:
			res.append([d[key],d_node[key],return_info[i]])
	if len(res)==0:
		return np.array([[np.nan, np.nan, np.nan]])
	res=np.array(res,dtype='object')
	srt=np.argsort(res[:,0])[::-1]
	res=res[srt]
	return res

	
	
def maximize(panel,direction,mp,args,channel,msg_main="",log=[]):
	"""Maxmizes logl.LL"""

	its, k, m, dx_norm,incr		=0,  [1,0,-1,0],     0,    None, 0
	H    = None
	msg,lmbda,lmbda0_count,rev='',	1,0,False
	direction.hessin_num, ll= None, None
	args_archive			= panel.input.args_archive
	reversed_direction=[]
	maximize_num2.maximize_test(panel,args.args_v)
	maximize_num.maximize_test(panel,args.args_v)
	ll=direction.init_ll(args)
	stpmax=100*max((abs(sum(ll.args.args_v**2)))**0.5,float(len(ll.args.args_v))) 
	po=printout(channel,panel,ll,direction,msg_main)
	min_iter=2#panel.options.minimum_iterations.value
	if not printout_func(0.0,'Determining direction',ll,its,direction,incr,po,0):return ll,direction,po
	#direction.start_time=time.time()
	while 1:
		prev_dx=direction.dx
		direction.calculate(ll,its,msg,incr,lmbda,sum(reversed_direction)>0)
		direction.set(mp,ll.args)
		LL0=ll.LL
		#Convergence test:
		conv=convergence_test(direction, its, args_archive, min_iter,ll,panel,lmbda,k,incr)
		if conv: 
			printout_func(1.0,"Convergence on zero gradient; maximum identified",ll,its,direction,incr,po,3,'done')
			return ll,direction,po
		if lmbda==0:
			printout_func(1.0,"No increase in linesearch without convergence. This should not happen. Contact Espen.Sirnes@uit.no",ll,its,direction,incr,po,1,"err")
		#if not printout_func(1.0,"",ll,its,direction,incr,po,1,task='linesearch'):return ll,direction,po
		
		
		#ll,msg,lmbda,ok,rev=lnsrch(ll,direction,mp,its,incr,po,prev_dx,direction.dx,max_its=5)
		fret,x,check, _ , lmbda=maximize_num.lnsrch(ll.args.args_v,-ll.LL,-direction.g,direction.dx,stpmax,direction.function.LL) 
		ll=direction.function.ll
		msg=''
		ok=True
		rev=False
		
		
		incr=ll.LL-LL0
		#log.append([incr,ll.LL,msg]+list(ll.args.args_v)+list(direction.dx_norm)+list(direction.g)+list(np.diag(direction.H))+list(direction.H[0]))
		#if not printout_func(1.0,msg,ll,its,direction,incr,po,0):return ll,direction,po
		print(f"{ll.LL}:{its}")
		if its==10:
			t=time.time()-direction.start_time
			a=0
		its+=1

		
def fast_max(H,g_old,dx,panel,direction,ll_orig):
	hessin=drctn.inv_hess(H)
	args=ll_orig.args.args_v
	ll_old=ll_orig.LL
	while 1:
		dx=-np.dot(hessin,g).flatten()
		args=args+dx
		ll=logl.LL(args,panel)
		direction.calc_gradient(ll)
		g_old=g
		g=direction.g
		hessin=drctn.nummerical_hessin(g,g_old,hessin,dx)
		

		

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
	
	
def convergence_test(direction,its,args_archive,min_iter,ll,panel,lmbda,k,incr):
	pgain=potential_gain(direction)
	#print(pgain)
	
	if incr>0:
		args_archive.save(ll.args,True)

	conv=(max(pgain)<panel.options.tolerance.value) and (max(np.abs(direction.dx_norm))<0.0001)
	conv*=(its>=min_iter)
	sing_problems=(direction.H_correl_problem or direction.singularity_problems)
	if sing_problems and conv and (lmbda>0):
		k[3]+=1
		if k[3]>3:
			return True
		else:
			return False
	k[3]=0	
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

		

			