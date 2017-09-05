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
	args=ll.args_d#using dictionary arguments
	if X is None:
		X=panel.X
	matrices=lgl.set_garch_arch(panel,args)
	if matrices is None:
		return None		

	AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
	(N,T,k)=panel.X.shape
	
	da=args['beta']*0
	da[0]=d
	u=panel.Y-fu.dot(panel.X,args['beta']+da)
	epsilon=(fu.dot(AMA_1AR,u))

	if panel.m>0:
		h_res=ll.h(epsilon, args['z'][0])
		if h_res==None:
			return None
		(h_val,h_epsilon_val,h_2epsilon_val,h_z_val,h_2z_val,h_epsilonz_val)=[i*panel.included for i in h_res]
		maxval=200/np.sum(h_val)
		lnv_ARMA=fu.dot(GAR_1MA,h_val)
	else:
		(h_val,h_epsilon_val,h_2epsilon_val,h_z_val,h_2z_val,h_epsilonz_val,avg_h)=(0,0,0,0,0,0,0)
		lnv_ARMA=0	
	if not panel.W_a is None:
		W_omega=fu.dot(panel.W_a,args['omega'])
	else:
		W_omega=0
	sigma=ll.args_d['sigma_e']+ll.args_d['sigma_v']
	ln_sigma=np.log(sigma)		
	lnv=ln_sigma+W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h=panel.mean(h_val,1).reshape((N,1,1))*panel.a
		if panel.N>1:
			lnv=lnv+args['mu'][0]*avg_h

	v=sigma#np.exp(lnv)*panel.a
	v_inv=1/v#np.exp(-lnv)*panel.a	
	eta=(epsilon)*v_inv**0.5
	e=ll.re_obj.RE(ll,(epsilon),eta,v)
	eta_mean=panel.mean(eta,1).reshape(N,1,1)
	e2=((epsilon)-(v**0.5)*eta_mean)*panel.included		
	e_sq=e**2
	e_st_sq=e_sq*v_inv

	LL=ll.LL_const-0.5*np.sum((np.log(v)+e_st_sq)*panel.included)

	return LL,e,e2,epsilon,np.log(v)


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