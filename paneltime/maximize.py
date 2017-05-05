#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import statproc as stat

digits_precision=7


def lnsrch(f0, g, dx, panel):
	
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
	for i in range(15+len(dx)):#Setting lmda so that the largest step is valid. Set panel.LL to return None when input is invalid
		lmda=0.5**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f1=panel.LL(x)
		if f1 is not None:
			if f1.LL is not None: 
				break
	if i==14+len(x):
		return f0
	f05=None
	i=0
	while f05 is None:
		i+=1
		f05=panel.LL(f0.args_v+lmda*(0.5**i)*dx)
	d={f1.LL:f1,f05.LL:f05}
	if f1.LL>f05.LL and f1.LL>f0.LL:
		return f1
	lmda_pred=lmda
	if ((f1.LL-f0.LL)*(f1.LL-f05.LL)>0) and (f05.LL!=f0.LL):
		lmda_pred = lmda*0.5*((f1.LL-f0.LL)+((f1.LL-f0.LL)*(f1.LL-f05.LL))**0.5)/(f05.LL-f0.LL)
		lmda_pred = max((min((lmda_pred,lmda)),0.1))
		f_lmda_pred=panel.LL(f0.args_v+lmda_pred*dx) 
		if not f_lmda_pred is None:
			d[f_lmda_pred.LL]=f_lmda_pred
	f_max=max(d.keys())
	if f_max<=f0.LL:#the function has not increased
		for j in range(1,6):
			lmda=lmda_pred*(0.05**j)
			ll=panel.LL(f0.args_v+lmda*dx) 
			if ll is None:
				break
			if ll.LL>f0.LL:
				return ll
	else:
		return d[f_max]
	return f0#should never happen

				

def maximize(panel,args=None,_print=True):
	"""Maxmizes panel.LL"""
	ll=panel.LL(args)
	if ll is None:
		print("""You requested stored arguments from a previous session 
		to be used as initial arguments (loadargs=True) but these failed to 
		return a valid log likelihood with the new parameters. Default inital 
		arguments will be used. """)
		ll=panel.LL(panel.args.start_args)
	its=0
	mc_limit_init=300
	mc_limit_min=0
	mc_limit=mc_limit_init
	convergence_limit=0.01
	has_problems=False
	k=0
	dx_conv=None
	while 1:  
		its+=1
		dx,g,G,H,constrained,reset=panel.get_direction(ll,mc_limit,mc_limit!=mc_limit_init,dx_conv,has_problems,k,its)
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

def printout(_print,ll,dx_conv,panel):
	ll.standardize(panel)
	norm_prob=stat.JB_normality_test(ll.e_st,panel)	
	if _print: 
		print("LL: %s Normality probability: %s " %(ll.LL,norm_prob))
		print("New direction in %% of argument: \n%s" %(np.round(dx_conv*100,2),))	
		print("Coefficients : \n%s" %(ll.args_v,))	
		
		
pass