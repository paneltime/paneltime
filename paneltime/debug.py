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
	"""Calculate the hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debuggeed gradient"""
	x=ll.args_v
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
	x=ll.args_v
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
				g[i]=(f1.LL-f0.LL)/dxi[i]
				break
	return g




def grad_debug_allparams(f0,panel,d,varname,pos=0):
	args=fu.copy_array_dict(f0.args_d)
	args[varname][pos]+=d
	f1=lgl.LL(args, panel)
	dLL=(f1.LL-f0.LL)/d
	
	for i in f1.__dict__.keys():
		x0=f0.__dict__[i]
		x1=f1.__dict__[i]
		if (type(x1)==np.ndarray):
			print(i)
			print((np.sum(x1-x0)/d))
	#LL_calc(f0, panel, d)
	a=0
	
	
def grad_debug_allparams_diff(f0,panel):

	f1=lgl.LL(f0.args_d, panel)
	for i in f1.__dict__.keys():
		x0=f0.__dict__[i]
		x1=f1.__dict__[i]
		if (type(x1)==np.ndarray):
			print(i)
			print((np.sum(x1-x0)))
	#LL_calc(f0, panel, d)
	a=0
	
def grad_debug_detail(f0,panel,d,llname,varname1,pos1=0):
	args1=fu.copy_array_dict(f0.args_d)
	args1[varname1][pos1]+=d
	
	f0=lgl.LL(f0.args_d, panel)
	f1=lgl.LL(args1, panel)

	if type(llname)==list:
		ddL=(f1.__dict__[llname[0]].__dict__[llname[1]]-f0.__dict__[llname[0]].__dict__[llname[1]])/d
	else:
		ddL=(f1.__dict__[llname]-f0.__dict__[llname])/d
	return ddL


def grad_debug_detail_oldll(f0,panel,d,llname,varname1,pos1=0):
	args1=fu.copy_array_dict(f0.args_d)
	args1[varname1][pos1]+=d
	

	f1=lgl.LL(args1, panel)

	if type(llname)==list:
		ddL=(f1.__dict__[llname[0]].__dict__[llname[1]]-f0.__dict__[llname[0]].__dict__[llname[1]])
	else:
		ddL=(f1.__dict__[llname]-f0.__dict__[llname])
	return ddL
	
	
def hess_debug_detail(f0,panel,d,llname,varname1,varname2,pos1=0,pos2=0):
	args1=fu.copy_array_dict(f0.args_d)
	args2=fu.copy_array_dict(f0.args_d)
	args3=fu.copy_array_dict(f0.args_d)
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
	
	
def LL_calc2(ll,panel,d,X=None):
	self=ll
	args=self.args_d#using dictionary arguments
	args['beta'][3]+=d
	if X is None:
		X=panel.X
	matrices=lgl.set_garch_arch(panel,args)
	if matrices is None:
		return None		

	AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
	(N,T,k)=X.shape

	u = panel.Y-cf.dot(X,args['beta'])
	e = cf.dot(AMA_1AR,u)
	lnv_ARMA = self.garch(panel, args, GAR_1MA,e)
	W_omega = cf.dot(panel.W_a,args['omega'])
	lnv = W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	grp = self.group_variance(panel, lnv, e,args)
	lnv+=grp
	lnv = np.maximum(np.minimum(lnv,100),-100)
	v = np.exp(lnv)*panel.a
	v_inv = np.exp(-lnv)*panel.a	
	e_RE = self.re_obj.RE(e)
	return e_RE
	e_REsq = e_RE**2
	LL = self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
	
	if abs(LL)>1e+100: 
		return None
	self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
	self.u,self.e, self.lnv_ARMA        = u,         e,       lnv_ARMA
	self.lnv,self.v,self.v_inv          = lnv,       v,       v_inv
	self.e_RE,self.e_REsq               = e_RE,      e_REsq

	return LL

def LL_calc(ll,panel,d,X=None):
	self=ll
	args=self.args_d#using dictionary arguments
	if X is None:
		X=panel.X
	matrices=lgl.set_garch_arch(panel,args)
	if matrices is None:
		return None		

	AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
	(N,T,k)=X.shape

	u = panel.Y-cf.dot(X,args['beta'])
	e = cf.dot(AMA_1AR,u)
	lnv_ARMA = self.garch(panel, args, GAR_1MA,e)
	W_omega = cf.dot(panel.W_a,args['omega'])
	lnv = W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	grp = self.group_variance(panel, lnv, e,args)
	lnv+=grp
	lnv = np.maximum(np.minimum(lnv,100),-100)
	v = np.exp(lnv)*panel.a
	v_inv = np.exp(-lnv)*panel.a	
	e_RE = self.re_obj.RE(e)+d*panel.included
	e_REsq = e_RE**2
	return -0.5*((lnv+(e_REsq)*v_inv)*panel.included)
	LL = self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
	
	if abs(LL)>1e+100: 
		return None
	self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
	self.u,self.e, self.lnv_ARMA        = u,         e,       lnv_ARMA
	self.lnv,self.v,self.v_inv          = lnv,       v,       v_inv
	self.e_RE,self.e_REsq               = e_RE,      e_REsq

	return LL

def LL_calc_custom(ll,panel,d,X=None):
	f0=LL_calc(ll,panel,0,X=None)
	f1=LL_calc(ll,panel,d,X=None)
	dLLeRE=(f1-f0)/d
	
	f0=LL_calc2(ll,panel,0,X=None)
	f1=LL_calc2(ll,panel,d,X=None)
	deRE=(f1-f0)/d
	return dLLeRE,deRE

	

def LL_calc_debug(ll,panel,g,d):
	f0=LL_calc(ll, panel,0)
	f1=LL_calc(ll, panel,d)
	f2=LL_calc(ll, panel,d*2)
	d_x=[]
	for i in range(5):
		d_x.append((f1[i]-f0[i])/d)
		print (np.sum(d_x[i]))

	dLL1=np.sum(g.DLL_e*d_x[1])
	dLL2=np.sum(g.DLL_e*d_x[2])
	dLL3=np.sum(g.DLL_e*d_x[3])
	
	#dd=np.sum(f2[2]-2*f1[2]+f0[2])/(d**2)
	a=0
	
	
