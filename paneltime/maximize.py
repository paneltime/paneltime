#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import loglikelihood as logl
import output
import sys
from tkinter import _tkinter
import time




def lnsrch(ll, direction,mp,its,incr,po):
	rmsg=''
	args=ll.args.args_v
	g = direction.g
	dx = direction.dx
	panel=direction.panel
	LL0=ll.LL
	constr=direction.constr
	
	
	if np.sum(g*dx)<0:
		dx=-dx
		rmsg="convex function at evaluation point, direction reversed - "	
	ll,msg,lmbda,ok=lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0,ll)
	if  ll.LL/panel.NT<-1e+15:
		msg='The maximization agorithm has gone mad. Resetting the argument to initial values'
		ll=logl.LL(panel.args.args_restricted,panel)
		return ll,msg,0,False
	if not ok and False:
		if not direction.progress_bar(0.95,msg):return ll,msg,lmbda,ok	
		i=np.argsort(np.abs(dx))[-1]
		dx=-ll.args.args_v*(np.arange(len(g))==i)
		ll,msg,lmbda,ok=lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0)
		if not ok:
			dx=ll.args.args_v*(np.arange(len(g))==i)
			ll,msg,lmbda,ok=lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0)			
	return ll,msg,lmbda,ok
		
		
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
	
	
def lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0,ll):
	mp.send_dict_by_file({'constr':constr})
	start=0
	end=2.0
	msg=''
	single_core=False#single_core=True has been tested, and even with a relatively small sample, multicore is much faster.
	for i in range(4):
		delta=(end-start)/(mp.master.cpu_count-1)
		res=get_likelihoods(args, dx, panel, constr, mp,delta,start,single_core)
		if i==0:
			res0=res[0]		
		if (res[0,1]==0 and i==0) or (res[0,0]<=res0[0] and i>0) or np.isnan(res[0,0]):#best was no change
			start=delta/mp.master.cpu_count
			end=delta
		else:
			if i>0:
				msg=f'Found increment at {round(res[0,2],4)} of Newton step'
			break
	if i>0:
		if np.max(res[:,0])==res0[0]:
			return ll,rmsg+'No increase in linesearch',0,False
		res=np.append([res0],res,0)#ensuring the original is in the LL set
		srt=np.argsort(res[:,0])[::-1]
		res=res[srt]
	res=remove_nan(res)
	if LL0>res[0,0]:
		print('Best linsearch is poorer than the starting point. You may have discovered a bug, please notify espen.sirnes@uit.no')
		ll,rmsg+'Best linsearch is poorer than the starting point. You may have discovered a bug, please notify espen.sirnes@uit.no',res[0,0]-LL0,False
	try:
		lmda=solve_square_func(res[0,0], res[0,2],res[1,0], res[1,2],res[2,0], res[2,2],res[0,2])
		ll=logl.LL(args+lmda*dx,panel,constr)
		if ll.LL<res[0,0]:
			raise RuntimeError('Something wrong with ll. You may have discovered a bug, please notify espen.sirnes@uit.no')
	except:
		ll, lmda = mp.remote_recieve(f'f{res[0,1]}',res[0,1],single_core), res[0,2]
	if lmda==0:
		return ll,rmsg+'No increase in linesearch',0,False
	if msg=='':
		msg=f"Linesearch success ({round(lmda,6)} of Newton step)"
	return ll,rmsg+msg,lmda,True

def remove_nan(res):
	r=[]
	for i in res:
		if not np.isnan(i[0]):
			r.append(i)
	return np.array(r)	
	
				
def get_likelihoods(args, dx,panel,constr,mp,delta,start,single_core):
	expr=[]	
	lamdas=[]
	args_lst=[]
	ids=range(mp.master.cpu_count)
	for i in ids:
		lmda=start+i*delta
		a=list(args+lmda*dx)
		lamdas.append(lmda)
		args_lst.append(a)
		expr.append([f"""
try:
	f{i}=lgl.LL({a}, panel,constr)
	LL{i}=f{i}.LL
except:
	LL{i}=None
""", f'LL{i}'])
	loca_dict={'panel':panel,'lgl':logl,'constr':constr}
	d=mp.remote_execute(expr,loca_dict,single_core) #dict only used for single_core
	res=[]
	for i in ids:
		if not d[f'LL{i}'] is None:
			res.append([d[f'LL{i}'],i,lamdas[i]])
	if len(res)==0:
		return np.array([[np.nan, np.nan, np.nan]])
	res=np.array(res,dtype='object')
	srt=np.argsort(res[:,0])[::-1]
	res=res[srt]
	return res
	

	
	
def maximize(panel,direction,mp,args,tab):
	"""Maxmizes logl.LL"""

	convergence_limit=panel.settings.convergence_limit.value[0]
	its, k, m, dx_norm,incr		=0,  1,     0,    None, 0
	H,  digits_precision    = None, 12
	msg,lmbda='',	1
	direction.hessin_num, ll= None, None
	args_archive			= panel.input.args_archive
	ll=direction.init_ll(args)
	po=printout(tab,panel,ll,direction)
	min_iter=panel.settings.minimum_iterations.value
	if not printout_func(0.0,'Determining direction',ll,its,direction,incr,po,0):return ll,direction,po
	b=0
	while 1:
		direction.get(ll,its,msg)
		LL0=ll.LL
			
		#Convergence test:
		conv,k=convergence_test(direction, convergence_limit, its, args_archive, min_iter, b,ll,panel,lmbda,k)
		if conv: 
			printout_func(1.0,"Convergence on zero gradient; maximum identified",ll,its,direction,incr,po,3)
			return ll,direction,po
		if lmbda==0:
			printout_func(1.0,"No increase in lnesearch without convergence. This should not happen. Contact Espen.Sirnes@uit.no",ll,its,direction,incr,po,1)
		if not printout_func(0.95,"Linesearch",ll,its,direction,incr,po,1):return ll,direction,po
		ll,msg,lmbda,ok=lnsrch(ll,direction,mp,its,incr,po)
		
		incr=ll.LL-LL0
		if not printout_func(1.0,msg,ll,its,direction,incr,po,0):return ll,direction,po
		its+=1
		
def convergence_test(direction,convergence_limit,its,args_archive,min_iter,b,ll,panel,lmbda,k):
	constr_conv=np.max(np.abs(direction.dx_norm))<convergence_limit*k
	unconstr_conv=np.max(np.abs(direction.dx_unconstr)) < convergence_limit*k
	conv=constr_conv or (unconstr_conv and its>3)
	args_archive.save(ll.args,conv)
	if lmbda==0:
		k=min((k*2,100))
		return conv,k
	k=1
	conv*=(its>=min_iter)
	if hasattr(direction,'HG_ratio'):
		conv*=(direction.HG_ratio>0)
	return conv,k
		
def printout_func(percent,msg,ll,its,direction,incr,po,update_type):
	po.printout(ll,its+1,direction,incr,update_type)	
	return direction.progress_bar(percent,msg)
		


	
class printout:
	def __init__(self,tab,panel,ll,direction,_print=True):
		self._print=_print
		self.tab=tab
		if tab is None:
			return
		tab.set_output_obj(ll, direction)

		
		
	def printout(self,ll, its, direction,incr,update_type):
		if self.tab is None and self._print:
			print(ll.LL)
			return
		if not self._print:
			return
		if update_type>0:
			self.tab.update_after_direction(direction,its)
		elif update_type==0 or update_type==2:
			self.tab.update_after_linesearch(direction,ll,incr)

		

			