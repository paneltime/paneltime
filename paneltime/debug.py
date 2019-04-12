#!/usr/bin/env python
# -*- coding: utf-8 -*-

#used for debugging
import numpy as np
import statproc as stat
import regprocs as rp
import time
import functions as fu
import os
import loglikelihood as lgl



def hess_debug(ll,panel,g,d):
	"""Calculate the hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debuggeed gradient"""
	x=ll.args_v
	n=len(x)
	dx=np.abs(x.reshape(n,1))*d
	dx=dx+(dx==0)*d
	dx=np.identity(n)*dx
	H=np.zeros((n,n))
	ll=lgl.LL(x,panel)
	f0=g.get(ll)
	for i in range(n):
		for j in range(5):
			dxi=dx[i]*(0.5**j)		
			ll=lgl.LL(x+dxi,panel)
			if not ll is None:
				f1=g.get(ll)
				H[i]=(f1-f0)/dxi[i]
				break
			
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




def grad_debug_detail(f0,panel,g,d,varname,pos=0):
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
	
	
def hess_debug_detail(f0,panel,g,H,d,varname1,varname2,pos1=0,pos2=0):
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
	ddL=(f3.LL-f2.LL-f1.LL+f0.LL)/(d**2)
	a=0
	

def LL_calc(ll,panel,d,X=None):
	self=ll
	args=self.args_d#using dictionary arguments
	if X is None:
		X=panel.X
	matrices=set_garch_arch(panel,args)
	if matrices is None:
		return None		

	AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
	(N,T,k)=panel.X.shape

	u=panel.Y-fu.dot(panel.X,args['beta'])
	e=fu.dot(AMA_1AR,u)

	if panel.m>0:
		h_res=self.h(e, args['z'][0])
		if h_res==None:
			return None
		(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val)=[i*panel.included for i in h_res]
		lnv_ARMA=fu.dot(GAR_1MA,h_val)
	else:
		(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val,avg_h)=(0,0,0,0,0,0,0)
		lnv_ARMA=0	

	W_omega=fu.dot(panel.W_a,args['omega'])
	lnv=W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_e2=panel.mean(e**2,1).reshape((N,1,1))
		avg_lne2=np.log(avg_e2)
		if panel.N>1:
			lnv=lnv+args['mu'][0]*avg_lne2*panel.a
		lnv=np.maximum(np.minimum(lnv,100),-100)
	v=np.exp(lnv)*panel.a
	v_inv=np.exp(-lnv)*panel.a	
	e_RE=self.re_obj.RE(e)
	e_REsq=e_RE**2
	LL=self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)

	if abs(LL)>1e+100: 
		return None
	self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
	self.u,self.e,self.h_e_val,self.h_val, self.lnv_ARMA        = u,e,h_e_val,h_val, lnv_ARMA
	self.lnv,self.avg_e2,self.v,self.v_inv,self.e_RE,self.e_REsq = lnv,avg_e2,v,v_inv,e_RE,e_REsq
	self.h_2e_val,self.h_z_val,self.h_ez_val,self.h_2z_val      = h_2e_val,h_z_val,h_ez_val,h_2z_val
	self.e_st,self.avg_lne2=e_RE*v_inv,avg_lne2

	return LL
	
	a=0


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