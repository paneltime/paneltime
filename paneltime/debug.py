#!/usr/bin/env python
# -*- coding: utf-8 -*-

#used for debugging
import numpy as np
import stat_functions as stat
import calculus_functions as cf
import time
import functions as fu
import os
import loglikelihood as lgl



def hess_debug(ll,panel,g,d):
	"""Calculate the Hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debugged gradient"""
	x=ll.args.args_v
	n=len(x)
	dx=np.identity(n)*d
	H=np.zeros((n,n))
	ll0=lgl.LL(x,panel)
	f0=g.get(ll0)
	for i in range(n):
		ll=lgl.LL(x+dx[i],panel)
		if not ll is None:
			f1=g.get(ll)
			H[i]=(f1-f0)/d

			
	return H

def grad_debug(ll,panel,d):
	"""Calcualtes the gradient numerically. For debugging"""
	x=ll.args.args_v
	n=len(x)
	dx=np.abs(x.reshape(n,1))*d
	dx=dx+(dx==0)*d
	dx=np.identity(n)*dx

	g=np.zeros(n)
	f0=lgl.LL(x,panel)
	for i in range(n):
		for j in range(5):
			dxi=dx[i]*(0.5**j)
			f1=lgl.LL(x+dxi,panel)
			if not f1 is None:
				if not f1.LL is None:
					g[i]=(f1.LL-f0.LL)/dxi[i]
					break
	return g


	
def grad_debug_detail(f0,panel,d,llname,varname1,pos1=0):
	args1=fu.copy_array_dict(f0.args.args_d)
	args1[varname1][pos1]+=d
	
	f0=lgl.LL(f0.args.args_d, panel)
	f1=lgl.LL(args1, panel)

	if type(llname)==list or type(llname)==tuple:
		ddL=(f1.__dict__[llname[0]].__dict__[llname[1]]-f0.__dict__[llname[0]].__dict__[llname[1]])/d
	else:
		ddL=(f1.__dict__[llname]-f0.__dict__[llname])/d
	return ddL


	
def hess_debug_detail(f0,panel,d,llname,varname1,varname2,pos1=0,pos2=0):
	args1=fu.copy_array_dict(f0.args.args_d)
	args2=fu.copy_array_dict(f0.args.args_d)
	args3=fu.copy_array_dict(f0.args.args_d)
	args1[varname1][pos1]+=d
	args2[varname2][pos2]+=d	
	args3[varname1][pos1]+=d
	args3[varname2][pos2]+=d
	f1=lgl.LL(args1, panel)
	f2=lgl.LL(args2, panel)
	f3=lgl.LL(args3, panel)
	if type(llname)==list:
		ddL=(f3.__dict__[llname[0]].__dict__[llname[1]]-f2.__dict__[llname[0]].__dict__[llname[1]]
		     -f1.__dict__[llname[0]].__dict__[llname[1]]+f0.__dict__[llname[0]].__dict__[llname[1]])/(d**2)
	else:
		ddL=(f3.__dict__[llname]-f2.__dict__[llname]-f1.__dict__[llname]+f0.__dict__[llname])/(d**2)
	return ddL
	


def LL_calc(self,panel):
	panel=self.panel
	X=panel.XIV
	matrices=set_garch_arch(panel,self.args.args_d)
	if matrices is None:
		return None		
	
	AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
	(N,T,k)=X.shape
	#Idea for IV: calculate Z*u throughout. Mazimize total sum of LL. 
	u = panel.Y-cf.dot(X,self.args.args_d['beta'])
	e = cf.dot(AMA_1AR,u)
	e_RE = (e+self.re_obj_i.RE(e, panel)+self.re_obj_t.RE(e, panel))*panel.included[3]
	
	e_REsq =(e_RE**2+(e_RE==0)*1e-18) 
	grp = self.variance_RE(panel,e_REsq)#experimental
	
	W_omega = cf.dot(panel.W_a, self.args.args_d['omega'])
	if panel.options.RE_in_GARCH.value:
		lnv_ARMA = self.garch(panel, GAR_1MA, e_RE)
	else:
		lnv_ARMA = self.garch(panel, GAR_1MA, e)	
	lnv = W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	lnv+=grp
	self.dlnv_pos=(lnv<100)*(lnv>-100)
	lnv = np.maximum(np.minimum(lnv,100),-100)
	v = np.exp(lnv)*panel.a[3]
	v_inv = np.exp(-lnv)*panel.a[3]

	LL = self.LL_const-0.5*(lnv+(e_REsq)*v_inv)

	self.tobit(panel,LL)
	LL=np.sum(LL*panel.included[3])
			
	self.add_variables(matrices, u, e, lnv_ARMA, lnv, v, W_omega, grp,e_RE,e_REsq,v_inv)
	if abs(LL)>1e+100: 
		return None				
	return LL



	
	
def pedantic(ll,msg,lmbda,ok,rev,direction,mp,its,incr,po,prev_dx,LL0,diff_log,ll0,dxLL,panel):
	"""Used in maximize.maximize() when the LL maximization does not converge. It is usually 
	easier to identify errors in gradient or hessian close to the maximum."""
	ll,msg2,lmbda2,ok,rev=lnsrch(ll,direction,mp,its,incr,po,prev_dx,direction.dx,max_its=6)
	dx=direction.dx
	n=len(dx)
	dx_norm=direction.dx_norm
	h=np.diag(direction.H)
	g=direction.g
	
	if panel.options.pedantic.value==1:
		return ll,msg,lmbda,ok,rev
	reversed_direction=g*dx<0

	for sgn in [1,-1]:
		if ll.LL-LL0<1e-6*len(dx) and sum(reversed_direction)>0:

			max_arg=((dxLL)==max((dxLL[reversed_direction])))
			for sel,d,s in [
				(max_arg==False,			  dx, 'directions other than the maxiumum'),
				(max_arg==True,				  dx, 'the maxiumum'),				
				(reversed_direction,	      dx, 'reversed directions'),
				(reversed_direction==False,	  dx, 'directions other than the reversed and reversing the reversed directions')
				]:
				dxi=sgn*np.array(d)
				dxi[sel]=0
				ll2,msg,lmbda2,ok,rev=lnsrch(ll,direction,mp,its,incr,po,prev_dx,dxi,max_its=6,convex_action='ignore')
				if ll2.LL>ll.LL:
					if not printout_func(1.0,f"Improved {ll2.LL-ll.LL} after fixing {s}",ll,its,direction,incr,po,1,task='linesearch'):return ll,direction,po
					lmbda+=lmbda2
					ll=ll2
					break

	if ll.LL-LL0<1e-6*len(dx):
		if len(diff_log)>2:
			avg_dir=np.mean(np.array(diff_log)[-8:],0)
			ll2,msg,lmbda2,ok,rev=lnsrch(ll,direction,mp,its,incr,po,prev_dx,avg_dir,max_its=6,convex_action='ignore')
			if ll2.LL>ll.LL:
				if not printout_func(1.0,f"Improved {ll2.LL-ll.LL} after following the average past directions",ll,its,direction,incr,po,1,task='linesearch'):return ll,direction,po
				ll=ll2
				lmbda+=lmbda2
	if ll.LL-LL0==0 and panel.options.pedantic.value==3:
		if not printout_func(1.0,f"Doing brute force",ll,its,direction,incr,po,1,task='linesearch'):return ll,direction,po
		ll,msg,lmbda2,ok,rev=brute_force(ll, mp, direction, its, incr, po, prev_dx, msg, lmbda, ok, 4)		
		lmbda+=lmbda2		
	diff_log.append(ll.args.args_v-ll0.args.args_v)
	
	return ll,msg,lmbda,ok,rev


		


def brute_force(ll,mp,direction,its,incr,po,prev_dx,msg,lmbda,ok,max_its):
	"""Brute force maximization for pedantic(), when everything else fails."""
	dx=direction.dx
	rng=np.arange(len(dx))
	for i in rng:
		for sign in [-1,1]:
			if not (i in direction.constr.fixed):
				ll2,msg,lmbda,ok,rev=lnsrch(ll,direction,mp,its,incr,po,prev_dx,sign*dx*(rng==i),max_its,convex_action='ignore')
				if ll2.LL>ll.LL:
					ll=ll2
					break
	return ll,msg,lmbda,ok,rev