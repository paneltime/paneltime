#!/usr/bin/env python
# -*- coding: utf-8 -*-

#simulalte a panel data GARCH model

import numpy as np

def  sim():
	pass
	

def generate_dataset(panel,N,T,beta,rho=[],lmbda=[],psi=[],gamma=[],omega=[],mu=[],z=[],residual_var=1,group_var=1):
	e=np.random.normal(0,1,(N,T))
	eRE=RE_errors(N,T,residual_var,group_var)
	args=panel.args.new_args(self,beta,rho,lmbda,psi,gamma,omega,mu,z)
	matrices=panel.set_garch_arch(args)
	if matrices is None:
		return None		
	AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices	
	lnv=
	#This should be inverted:
	args=self.args_d#using dictionary arguments
	

	(N,T,k)=panel.X.shape

	u=panel.Y-fu.dot(panel.X,args['beta'])
	e=fu.dot(AMA_1AR,u)

	if panel.m>0:
		h_res=rp.h_func(e, args['z'][0])
		if h_res==None:return None
		(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val)=[i*panel.included for i in h_res]
		lnv_ARMA=fu.dot(GAR_1MA,h_val)
	else:
		(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val,avg_h)=(0,0,0,0,0,0,0)
		lnv_ARMA=0	
	W_omega=fu.dot(panel.W_a,args['omega'])
	lnv=W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h=(np.sum(h_val,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv=lnv+args['mu'][0]*avg_h
	if np.any(np.abs(lnv)>700): return None
	v=np.exp(lnv)*panel.a
	v_inv=np.exp(-lnv)*panel.a	
	e_RE=rp.RE(self,panel,e)
	e_REsq=e_RE**2
	if center_e:
		e=e-np.mean(e)
	LL=self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
	
	
	
	
	
def RE_errors(N,T,e,residual_var,group_var=None):
	
	if not group_var is None:
		e=e+np.random.normal(0,1,(N,1))
	return e
	
	
	