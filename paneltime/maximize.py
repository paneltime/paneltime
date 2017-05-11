#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import statproc as stat
import loglikelihood as logl
import os

digits_precision=7


def lnsrch(f0, g, dx,panel):
	
	s=g*dx
	slope=np.sum(s)					#ensuring positive slope (should not be negative unless errors in gradient and/or hessian)
	if slope <= 0.0:
		print("Warning: Roundoff problem in lnsrch")
		sel=dx*np.sign(g)==max(dx*np.sign(g))
		g=sel*g
		dx=sel*dx
		slope=np.sum(g*dx)
	if slope<=0:
		return f0
	for i in range(15+len(dx)):#Setting lmda so that the largest step is valid. Set ll.LL to return None when input is invalid
		lmda=0.5**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f1=logl.LL(x,panel)
		if f1.LL is not None:
			break
	if i==14+len(x):
		return f0
	i=0
	while 1:
		i+=1
		f05=logl.LL(f0.args_v+lmda*(0.5**i)*dx,panel)
		if not f05.LL is None:
			break
	d={f1.LL:f1,f05.LL:f05}
	if f1.LL>f05.LL and f1.LL>f0.LL:
		return f1
	lmda_pred=lmda
	if ((f1.LL-f0.LL)*(f1.LL-f05.LL)>0) and (f05.LL!=f0.LL):
		lmda_pred = lmda*0.5*((f1.LL-f0.LL)+((f1.LL-f0.LL)*(f1.LL-f05.LL))**0.5)/(f05.LL-f0.LL)
		lmda_pred = max((min((lmda_pred,lmda)),0.1))
		f_lmda_pred=logl.LL(f0.args_v+lmda_pred*dx,panel) 
		if not f_lmda_pred.LL is None:
			d[f_lmda_pred.LL]=f_lmda_pred
	f_max=max(d.keys())
	if f_max<=f0.LL:#the function has not increased
		for j in range(1,6):
			lmda=lmda_pred*(0.05**j)
			ll=logl.LL(f0.args_v+lmda*dx,panel) 
			if ll.LL is None:
				break
			if ll.LL>f0.LL:
				return ll
	else:
		return d[f_max]
	return f0#should never happen

				

def maximize(panel,direction,mp,args=None,_print=True):
	"""Maxmizes logl.LL"""
	
	
	ll=logl.LL(args,panel)
	if ll.LL is None:
		print("""You requested stored arguments from a previous session 
		to be used as initial arguments (loadargs=True) but these failed to 
		return a valid log likelihood with the new parameters. Default inital 
		arguments will be used. """)
		ll=logl.LL(panel.args.start_args,panel)
	its=0
	mc_limit_init=300
	mc_limit_min=0
	mc_limit=mc_limit_init
	convergence_limit=0.01
	has_problems=False
	k=0
	dx_conv=None
	H=None
	ll=pretest(ll, panel,direction,mp)
	while 1:  
		its+=1
		dx,g,G,H,constrained,reset=direction.get(ll,mc_limit,mc_limit!=mc_limit_init,dx_conv,k,its,H,mp)
		if reset:
			k=0
		f0=ll
		LL0=round_sign(f0.LL,digits_precision)
		dx_conv=(ll.args_v!=0)*np.abs(dx)*(constrained==0)/(np.abs(ll.args_v)+(ll.args_v==0))
		dx_conv=(ll.args_v==0)*dx+dx_conv
		printout(_print, ll, dx_conv,panel)
		#Convergence test:
		if np.max(dx_conv) < convergence_limit and (its>3 or  np.sum(constrained)<=2):  #max direction smaller than convergence_limit -> covergence
			if _print: print("Convergence on zero gradient; maximum identified")
			return ll,g,G,H,1
		ll=lnsrch(ll,g,dx,panel) 
		panel.args_bank.save(ll.args_d,0)
		

		test=np.max(np.abs(f0.args_v-ll.args_v)/np.maximum(np.abs(ll.args_v),1e-50))
		if round_sign(ll.LL,digits_precision)==LL0 or (test < 12.0e-16 ):#happens when the functions has not increased or arguments not changed
			if np.sum(constrained)>=len(constrained):
				print("Unable to reach convergence")
				return ll,g,G,H, 0 
			if mc_limit==mc_limit_min:
				mc_limit=mc_limit_init#increases the number of restricted variables
			else:
				mc_limit=mc_limit_min
				k+=1
			has_problems=True
		else:
			mc_limit=mc_limit_init
			has_problems=False
			k=0
	

def round_sign(x,n):
	"""rounds to n significant digits"""
	return round(x, -int(np.log10(abs(x)))+n-1)


def pretest(ll,panel,direction,mp):
	if mp is None:
		ll=pretest_sub(ll,['psi','gamma'],panel,direction)
		ll=pretest_sub(ll,['rho','lambda'],panel,direction)
		return ll
	else:
		ll=pretest_master(ll,mp)
	
	
def pretest_master(ll,mp):
	c=['psi','gamma','rho','lambda']
	k=len(c)
	a=np.array(np.meshgrid(*tuple([[-0.5, 0.5]]*k))).T.reshape(-1,k)
	n=len(a)
	args=[]
	for i in a:
		args_d=ll.copy_args_d()
		for j in range(k):
			args_d[c[j]][0]=i[j]
		args.append(args_d)
	expr=[]
	for i in range(4):
		expr.append('res%s=mx.pretest_func(panel,direction,args[%s])' %(i,i))
	mp.send_dict({'args':args},'dynamic dictionary')	
	d=mp.execute(expr)
	
	max_ll=ll
	for i in range(n):
		ll_new=d['res%s' %(i,)]
		print('pretest LL: %s  Max LL: %s' %(ll_new.LL,max_ll.LL))
		if not ll_new is None:
			if ll_new.LL>max_ll.LL:
				max_ll=ll_new	
	return max_ll
	
def pretest_func(panel,direction,args):
	ll_new=logl.LL(args,panel)
	if ll_new.LL is None:
		return None
	dx,g,G,H,constrained,reset=direction.get(ll_new,1000,False,None,0,0,None)
	ll_new=lnsrch(ll_new,g,dx,panel) 
	return ll_new
	
def pretest_sub(ll,categories,panel,direction):
	c=categories
	vals=[-0.5,0.5]
	max_ll=ll
	args_d=ll.copy_args_d()
	for i in vals:
		for j in vals:
			for k in [0,1]:
				if len(args_d[c[k]])>0:
					args_d[c[k]][0]=[i,j][k]
			ll_new=logl.LL(args_d,panel)
			dx,g,G,H,constrained,reset=direction.get(ll_new,1000,False,None,0,0,None)
			ll_new=lnsrch(ll_new,g,dx,panel) 
			print('pretest LL: %s  Max LL: %s' %(ll_new.LL,max_ll.LL))
			if ll_new.LL>max_ll.LL:
				max_ll=ll_new
	return max_ll

def printout(_print,ll,dx_conv,panel):
	ll.standardize(panel)
	norm_prob=stat.JB_normality_test(ll.e_st,panel)	
	if _print: 
		print("LL: %s Normality probability: %s " %(ll.LL,norm_prob))
		print("New direction in %% of argument: \n%s" %(np.round(dx_conv*100,2),))	
		print("Coefficients : \n%s" %(ll.args_v,))	
		
		
pass