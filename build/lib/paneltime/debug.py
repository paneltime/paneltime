#!/usr/bin/env python
# -*- coding: utf-8 -*-

#used for debugging
import numpy as np
import statproc as stat
import regprocs as rp
import time
import functions as fu
import os

def LL_debug_detail(ll,panel,d):
	"""Calcuates LL after some variable change with d. Used for debugging. Which variable is affected must be specified manually below"""
	args=ll.args_d
	(N,T,k)=panel.X.shape
	s=0
	if s==1: return
	
	#****LL0*****
	matrices=set_garch_arch_debug(panel,args,0)	
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	u=panel.Y-fu.dot(panel.X,beta_)
	e_0=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_0, args['z'])
		if h_res==None:return None
		(h_val_0,h_e_val_0,h_2e_val_0,h_c_val_0,h_2c_val_0,h_ec_val_0)=[i*panel.included for i in h_res]		
		lnv_ARMA_0=np.moveaxis(np.dot(GAR_1MA,h_val_0),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_0,h_e_val_0,h_2e_val_0,h_c_val_0,h_2c_val_0,h_ec_val_0,avg_h_0)=(0,0,0,0,0,0,0)
		lnv_ARMA_0=0	
	lnv_0=fu.dot(panel.W_a,args['omega'])+lnv_ARMA_0# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_0=(np.sum(h_val_0,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_0=lnv_0+args['mu']*avg_h_0		
	v_0=np.exp(lnv_0)*panel.a
	v_inv_0=np.exp(-lnv_0)*panel.a	
	e_RE_0=rp.RE(ll,panel,e_0)
	e_REsq_0=e_RE_0**2
	LL_value_0=ll.LL_const-0.5*np.sum((lnv_0+(e_REsq_0)*v_inv_0)*panel.included)	

	#****LL1*****
	matrices=set_garch_arch_debug(panel,args,d)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	u=panel.Y-fu.dot(panel.X,beta_)
	e_1=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_1, args['z'])
		if h_res==None:return None
		(h_val_1,h_e_val_1,h_2e_val_1,h_c_val_1,h_2c_val_1,h_ec_val_1)=[i*panel.included for i in h_res]		
		lnv_ARMA_1=np.moveaxis(np.dot(GAR_1MA,h_val_1),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_1,h_e_val_1,h_2e_val_1,h_c_val_1,h_2c_val_1,h_ec_val_1,avg_h_1)=(0,0,0,0,0,0,0)
		lnv_ARMA_1=0
	lnv_1=fu.dot(panel.W_a,args['omega'])+lnv_ARMA_1# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_1=(np.sum(h_val_1,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_1=lnv_1+args['mu']*avg_h_1	
	v_1=np.exp(lnv_1)*panel.a
	v_inv_1=np.exp(-lnv_1)*panel.a	
	e_RE_1=rp.RE(ll,panel,e_1)
	e_REsq_1=e_RE_1**2
	LL_value_1=ll.LL_const-0.5*np.sum((lnv_1+(e_REsq_1)*v_inv_1)*panel.included)	
	
	#****LL2*****
	matrices=set_garch_arch_debug(panel,args,0)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	beta_d=beta_*0
	beta_d[0][0]=d
	u=panel.Y-fu.dot(panel.X,beta_+beta_d)
	e_2=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_2, args['z'])
		if h_res==None:return None
		(h_val_2,h_e_val_2,h_2e_val_2,h_c_val_2,h_2c_val_2,h_ec_val_2)=[i*panel.included for i in h_res]		
		lnv_ARMA_2=np.moveaxis(np.dot(GAR_1MA,h_val_2),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_2,h_e_val_2,h_2e_val_2,h_c_val_2,h_2c_val_2,h_ec_val_2,avg_h_2)=(0,0,0,0,0,0,0)
		lnv_ARMA_2=0
	lnv_2=fu.dot(panel.W_a,args['omega'])+lnv_ARMA_2# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_2=(np.sum(h_val_2,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_2=lnv_2+args['mu']*avg_h_2
	v_2=np.exp(lnv_2)*panel.a
	v_inv_2=np.exp(-lnv_2)*panel.a	
	e_RE_2=rp.RE(ll,panel,e_2)
	e_REsq_2=e_RE_2**2
	LL_value_2=ll.LL_const-0.5*np.sum((lnv_2+(e_REsq_2)*v_inv_2)*panel.included)
	
	#****LL3*****
	matrices=set_garch_arch_debug(panel,args,d)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	beta_d=beta_*0
	beta_d[0][0]=d
	u=panel.Y-fu.dot(panel.X,beta_+beta_d)
	e_3=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_3, args['z'])
		if h_res==None:return None
		(h_val_3,h_e_val_3,h_3e_val_3,h_c_val_3,h_3c_val_3,h_ec_val_3)=[i*panel.included for i in h_res]		
		lnv_ARMA_3=np.moveaxis(np.dot(GAR_1MA,h_val_3),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_3,h_e_val_3,h_3e_val_3,h_c_val_3,h_3c_val_3,h_ec_val_3,avg_h_3)=(0,0,0,0,0,0,0)
		lnv_ARMA_3=0 
	lnv_3=fu.dot(panel.W_a,args['omega'])+lnv_ARMA_3# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_3=(np.sum(h_val_3,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_3=lnv_3+args['mu']*avg_h_3
	v_3=np.exp(lnv_3)*panel.a
	v_inv_3=np.exp(-lnv_3)*panel.a	
	e_RE_3=rp.RE(ll,panel,e_3)
	e_REsq_3=e_RE_3**2
	LL_value_3=ll.LL_const-0.5*np.sum((lnv_3+(e_REsq_3)*v_inv_3)*panel.included)
	
	
	#****derivatives*****	
	_val=LL_value_0
	d1_val=ll.LL_const-0.5*np.sum((lnv_0+(e_REsq_1)*v_inv_0)*panel.included)
	d2_val=ll.LL_const-0.5*np.sum((lnv_0+(e_REsq_2)*v_inv_0)*panel.included)
	dd_val=ll.LL_const-0.5*np.sum((lnv_0+(e_REsq_3)*v_inv_0)*panel.included)
	
	testd1_e=np.sum((d1_val-_val)*panel.included)/d
	testd2_e=np.sum((d2_val-_val)*panel.included)/d
	testdd_e=np.sum((dd_val-d1_val-d2_val+_val)*panel.included)/d**2
	
	_val=LL_value_0
	d1_val=ll.LL_const-0.5*np.sum((lnv_1+(e_REsq_0)*v_inv_1)*panel.included)
	d2_val=ll.LL_const-0.5*np.sum((lnv_2+(e_REsq_0)*v_inv_2)*panel.included)
	dd_val=ll.LL_const-0.5*np.sum((lnv_3+(e_REsq_0)*v_inv_3)*panel.included)
	
	testd1_v=np.sum((d1_val-_val)*panel.included)/d
	testd2_v=np.sum((d2_val-_val)*panel.included)/d
	testdd_v=np.sum((dd_val-d1_val-d2_val+_val)*panel.included)/d**2
	
	
	_val=e_RE_0
	d1_val=e_RE_1
	d2_val=e_RE_2
	dd_val=e_RE_3
	
	testd1=np.sum((d1_val-_val)*panel.included)/d
	testd2=np.sum((d2_val-_val)*panel.included)/d
	testdd=np.sum((dd_val-d1_val-d2_val+_val)*panel.included)/d**2
	
	_val=LL_value_0
	d1_val=LL_value_1
	d2_val=LL_value_2
	dd_val=LL_value_3	

	test1d=(d1_val-_val)/d
	test2d=(d2_val-_val)/d
	test2dd=(dd_val-d1_val-d2_val+_val)/d**2
	return testdd

def LL_debug_detail_single(panel,ll,d):
	c=ll.syncronize(panel)
	"""Calcuates LL after some variable change with d. Used for debugging. Which variable is affected must be specified manually below"""
	(N,T,k)=panel.X.shape
	s=0
	if s==1: return
	
	matrices=set_garch_arch_debug(panel,c,0)	
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	u=panel.Y-fu.dot(panel.X,beta_)
	e_0=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_0, c[panel.z_sel])
		if h_res==None:return None
		(h_val_0,h_e_val_0,h_2e_val_0,h_c_val_0,h_2c_val_0,h_ec_val_0)=[i*panel.included for i in h_res]		
		lnv_ARMA_0=np.moveaxis(np.dot(GAR_1MA,h_val_0),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_0,h_e_val_0,h_2e_val_0,h_c_val_0,h_2c_val_0,h_ec_val_0,avg_h_0)=(0,0,0,0,0,0,0)
		lnv_ARMA_0=0	
	lnv_0=fu.dot(panel.W_a,Wbeta_)+lnv_ARMA_0# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_0=(np.sum(h_val_0,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_0=lnv_0+c[panel.mu_sel]*avg_h_0		
	v_0=np.exp(lnv_0)*panel.a
	v_inv_0=np.exp(-lnv_0)*panel.a	
	e_RE_0=rp.RE(panel,e_0)
	e_REsq_0=e_RE_0**2
	LL_value_0=panel.LL_const-0.5*np.sum((lnv_0+(e_REsq_0)*v_inv_0)*panel.included)	

	matrices=set_garch_arch_debug(panel,c,0)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	u=panel.Y-fu.dot(panel.X,beta_)
	e_1=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_1, c[panel.z_sel]+d)
		if h_res==None:return None
		(h_val_1,h_e_val_1,h_2e_val_1,h_c_val_1,h_2c_val_1,h_ec_val_1)=[i*panel.included for i in h_res]		
		lnv_ARMA_1=np.moveaxis(np.dot(GAR_1MA,h_val_1),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_1,h_e_val_1,h_2e_val_1,h_c_val_1,h_2c_val_1,h_ec_val_1,avg_h_1)=(0,0,0,0,0,0,0)
		lnv_ARMA_1=0
	lnv_1=fu.dot(panel.W_a,Wbeta_)+lnv_ARMA_1# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_1=(np.sum(h_val_1,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_1=lnv_1+c[panel.mu_sel]*avg_h_1	
	v_1=np.exp(lnv_1)*panel.a
	v_inv_1=np.exp(-lnv_1)*panel.a	
	e_RE_1=rp.RE(panel,e_1)
	e_REsq_1=e_RE_1**2
	LL_value_1=panel.LL_const-0.5*np.sum((lnv_1+(e_REsq_1)*v_inv_1)*panel.included)	

	matrices=set_garch_arch_debug(panel,c,0)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	u=panel.Y-fu.dot(panel.X,beta_+np.array([[0],[0]]))
	e_2=np.moveaxis(np.dot(AMA_1AR,u),0,1)
	if panel.m>0:
		h_res=rp.h_func(e_2, c[panel.z_sel]+2*d)
		if h_res==None:return None
		(h_val,h_e_val,he_val,h_z_val,hc_val,h_ez_val)=[i*panel.included for i in h_res]		
		lnv_ARMA_2=np.moveaxis(np.dot(GAR_1MA,h_val_2),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	else:
		(h_val_2,h_e_val_2,h_2e_val_2,h_c_val_2,h_2c_val_2,h_ec_val_2,avg_h_2)=(0,0,0,0,0,0,0)
		lnv_ARMA_2=0
	lnv_2=fu.dot(panel.W_a,Wbeta_)+lnv_ARMA_2# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	if panel.m>0:
		avg_h_2=(np.sum(h_val_2,1)/panel.T_arr).reshape((N,1,1))*panel.a
		lnv_2=lnv_2+c[panel.mu_sel]*avg_h_2
	v_2=np.exp(lnv_2)*panel.a
	v_inv_2=np.exp(-lnv_2)*panel.a	
	e_RE_2=rp.RE(panel,e_2)
	e_REsq_2=e_RE_2**2
	LL_value_2=panel.LL_const-0.5*np.sum((lnv_2+(e_REsq_2)*v_inv_2)*panel.included)


	
	_val=h_val_0
	d1_val=h_val_1
	d2_val=h_val_2
	
	testd1=np.sum((d1_val-_val)*panel.included)/d
	testd2=np.sum((d2_val-d1_val)*panel.included)/d
	testdd=np.sum((d2_val-2*d1_val+_val)*panel.included)/d**2
	
	_val=LL_value_0
	d1_val=LL_value_1
	d2_val=LL_value_2

	test1d=(d1_val-_val)/d
	test2d=(d2_val-_val)/d
	test2dd=(d2_val-2*d1_val+_val)/d**2
	return testdd




def set_garch_arch_debug(self,args,d):

	p,q,m,k,nW=self.p,self.q,self.m,self.k,self.nW
	beta,rho,lambda_,gamma,psi,Wbeta=args['beta'],args['rho'],args['lambda'],args['gamma'],args['psi'],args['omega']
	beta_=beta.reshape((len(beta),1))
	Wbeta_=Wbeta.reshape((len(Wbeta),1))
	X=self.I+rp.lag_matr(self.L,self.zeros,q,lambda_)
	if not fu.cond_test(X):
		return None
	AMA_1=np.linalg.inv(X)
	AAR=self.I-rp.lag_matr(self.L,self.zeros,p,rho+d)
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=self.I-rp.lag_matr(self.L,self.zeros,k,gamma)
	if not fu.cond_test(X):
		return None
	GAR_1=np.linalg.inv(X)
	GMA=rp.lag_matr(self.L,self.zeros,m,psi)	
	GAR_1MA=fu.dot(GAR_1,GMA)

	return beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA

def hess_debug(panel,x,d):
	"""Calculate the hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debuggeed gradient"""

	n=len(x)
	dx=np.abs(x.reshape(n,1))*d
	dx=dx+(dx==0)*d
	dx=np.identity(n)*dx
	H=np.zeros((n,n))
	ll=panel.LL(x)
	f0=panel.gradient.get(ll)
	for i in range(n):
		for j in range(5):
			dxi=dx[i]*(0.5**j)		
			ll=panel.LL(x+dxi)
			if not ll is None:
				f1=panel.gradient.get(ll)
				H[i]=(f1-f0)/dxi[i]
				break
			
	return H

def grad_debug(panel,x,d):
	"""Calcualtes the gradient numerically. For debugging"""

	n=len(x)
	dx=np.abs(x.reshape(n,1))*d
	dx=dx+(dx==0)*d
	dx=np.identity(n)*dx

	g=np.zeros(n)
	f0=panel.LL(x)
	for i in range(n):
		for j in range(5):
			dxi=dx[i]*(0.5**j)
			f1=panel.LL(x+dxi)
			if not f1 is None:
				g[i]=(f1.LL-f0.LL)/dxi[i]
				break
	return g




def LL_debug(panel,ll,d):
	c=ll.syncronize(panel)*1
	"""Calcuates LL after some variable change with d. Used for debugging. Which variable is affected must be specified manually below"""
	c[panel.z_sel]=args_orig[panel.z_sel]+d#choose the variable to change

	matrices=panel.set_garch_arch(c)
	if matrices is None:
		return None		
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	(N,T,k)=panel.X.shape

	u=panel.Y-fu.dot(panel.X,beta_)
	e=np.moveaxis(fu.dot(AMA_1AR,u),0,1)
	h_res=rp.h_func(e, c[panel.z_sel])
	if h_res==None:return None
	(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val)=[i*panel.included for i in h_res]		
	lnv_ARMA=np.moveaxis(fu.dot(GAR_1MA,h_val),0,1)# 'T x T' * 'N x T x 1' -> 'T x N x 1' movaxis -> 'N x T x 1'
	lnv_W=fu.dot(panel.W_a,Wbeta_)+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
	avg_h=(np.sum(h_val,1)/panel.T_arr).reshape((N,1,1))*panel.a
	lnv=lnv_W+c[panel.mu_sel]*avg_h
	v=np.exp(lnv)*panel.a
	v_inv=np.exp(-lnv)*panel.a	
	v_inv_W=np.exp(-lnv_W)*panel.a	
	e_RE=RE(panel,e)
	e_REsq=e_RE**2
	LL_value=panel.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)

	return (u,e,h_e_val,h_val, lnv_ARMA,lnv,
            avg_h,v,v_inv,e_RE,e_REsq,LL_value,
            h_2e_val,h_z_val,h_ez_val,h_2z_val,
            AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA,lnv_W,v_inv_W)

def LL_hess_debug(panel,ll,d):
	"""Uses LL_dbug to debug the hessian in detail"""
	c=ll.syncronize(panel)
	(u,e,h_e_val,h_val, lnv_ARMA,lnv,
     avg_h,v,v_inv,e_RE,e_REsq,LL_value,
     h_2e_val,h_z_val,h_ez_val,h_2z_val,
     AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA,lnv_W,v_inv_W)=[3*[0] for i in range(24)]

	d_=[0,d,2*d]

	for i in range(3):
		(u[i],e[i],h_e_val[i],h_val[i], lnv_ARMA[i],lnv[i],
	     avg_h[i],v[i],v_inv[i],e_RE[i],e_REsq[i],LL_value[i],
	     h_2e_val[i],h_z_val[i],h_ez_val[i],h_2z_val[i],
	     AMA_1[i],AAR[i],AMA_1AR[i],GAR_1[i],GMA[i],GAR_1MA[i],lnv_W[i],v_inv_W[i])=panel.LL_debug(c,d_[i])	

	ddx,dx,x=panel.diffs_debug(LL_value[0],LL_value[1],LL_value[2],d)
	ddx2,dx2,x2=panel.diffs_debug(
        panel.LL_const-0.5*np.sum((lnv[0]+(e_REsq[0])*v_inv[0])*panel.included),
        panel.LL_const-0.5*np.sum((lnv[1]+(e_REsq[0])*v_inv[1])*panel.included),
        panel.LL_const-0.5*np.sum((lnv[2]+(e_REsq[0])*v_inv[2])*panel.included),
        d)
	ddx2,dx2,x2=panel.diffs_debug(
        panel.LL_const-0.5*np.sum((lnv_W[0]+(e_REsq[0])*v_inv_W[0])*panel.included),
        panel.LL_const-0.5*np.sum((lnv_W[1]+(e_REsq[0])*v_inv_W[1])*panel.included),
        panel.LL_const-0.5*np.sum((lnv_W[2]+(e_REsq[0])*v_inv_W[2])*panel.included),
        d)
	pass


def diffs_debug(panel,x0,x1,x2,d):
	ddx=np.sum(x2-2*x1+x0)/d**2
	dx=np.sum(x1-x0)/d
	x=np.sum(x0)
	return ddx,dx,x

def gradient(self,ll,return_G=False):
	u,e,h_e_val,lnv_ARMA,h_val,v,v_inv=ll.u,ll.e,ll.h_e_val,ll.lnv_ARMA,ll.h_val,ll.v,ll.v_inv
	p,d,q,m,k,nW=self.p,self.d,self.q,self.m,self.k,self.nW

	#ARIMA:
	de_rho=self.arima_grad(p,u,-1,ll.AMA_1)
	de_lambda=self.arima_grad(q,e,-1,ll.AMA_1)
	de_beta=-fu.dot(ll.AMA_1AR,self.X)
	(ll.de_rho,ll.de_lambda,ll.de_beta)=(de_rho,de_lambda,de_beta)

	ll.de_rho_RE,ll.de_lambda_RE,ll.de_beta_RE=rp.dRE(ll,self,de_rho,ll.e,'rho'),rp.dRE(ll,self,de_lambda,ll.e,'lambda'),rp.dRE(ll,self,de_beta,ll.e,'beta')
	#drRE=self.dLL(0.000001)


	dlnv_e_rho,		dlnv_e_rho_G	=	self.garch_arima_grad(ll,de_rho)
	dlnv_e_lambda, 	dlnv_e_lambda_G	=	self.garch_arima_grad(ll,de_lambda)
	dlnv_e_beta,	dlnv_e_beta_G	=	self.garch_arima_grad(ll,de_beta)

	(ll.dlnv_e_rho,ll.dlnv_e_lambda,ll.dlnv_e_beta)=(dlnv_e_rho,dlnv_e_lambda,dlnv_e_beta)
	(ll.dlnv_e_rho_G,ll.dlnv_e_lambda_G,ll.dlnv_e_beta_G)=(dlnv_e_rho_G,dlnv_e_lambda_G,dlnv_e_beta_G)

	#GARCH:
	if self.m>0:
		dlnv_gamma=self.arima_grad(k,lnv_ARMA,1,ll.GAR_1)
		dlnv_psi=self.arima_grad(m,h_val,1,ll.GAR_1)
		dlnv_z_G=fu.dot(ll.GAR_1MA,ll.h_z_val)
		(N,T,k)=dlnv_z_G.shape
		dlnv_z=dlnv_z_G+(ll.args_d['mu']*(np.sum(ll.h_z_val,1)/self.T_arr)).reshape(N,1,1)
		dlnv_mu=ll.avg_h
	else:
		(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)=(None,None,None,None,None)
	(ll.dlnv_gamma, ll.dlnv_psi,ll.dlnv_mu,ll.dlnv_z_G,ll.dlnv_z)=(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)

	#LL

	DLL_e=-(ll.e_RE*ll.v_inv)
	dLL_lnv=-0.5*(self.included-(ll.e_REsq*v_inv))
	(ll.DLL_e, ll.dLL_lnv)=(DLL_e, dLL_lnv)


	#final derivatives:
	dLL_beta=rp.add((rp.prod((dlnv_e_beta,dLL_lnv)),rp.prod((ll.de_beta_RE,DLL_e))),True)
	dLL_rho=rp.add((rp.prod((dlnv_e_rho,dLL_lnv)),rp.prod((ll.de_rho_RE,DLL_e))),True)
	dLL_lambda=rp.add((rp.prod((dlnv_e_lambda,dLL_lnv)),rp.prod((ll.de_lambda_RE,DLL_e))),True)
	dLL_gamma=rp.prod((dlnv_gamma,dLL_lnv))
	dLL_psi=rp.prod((dlnv_psi,dLL_lnv))
	dLL_omega=rp.prod((self.W_a,dLL_lnv))
	dLL_mu=rp.prod((ll.dlnv_mu,dLL_lnv))
	dLL_z=rp.prod((ll.dlnv_z,dLL_lnv))

	G=rp.concat_marray((dLL_beta,dLL_rho,dLL_lambda,dLL_gamma,dLL_psi,dLL_omega,dLL_mu,dLL_z))
	g=np.sum(np.sum(G,0),0)
	#gn=debug.grad_debug(self,ll.args_d,0.0000001)#debugging
	#self.dLL(0.01)
	#self.LL_debug_detail(ll.args_d,0.000001)
	if return_G:
		return  g,G
	else:
		return g

def hessian(self,ll):

	GARM=rp.ARMA_product(ll.GAR_1,self.L,self.m)
	GARK=rp.ARMA_product(ll.GAR_1,self.L,self.k)

	d2lnv_gamma2		=   rp.dd_func_lags(self,ll,GARK, 	ll.dlnv_gamma,						ll.dLL_lnv,  transpose=True)
	d2lnv_gamma_psi		=	rp.dd_func_lags(self,ll,GARK, 	ll.dlnv_psi,						ll.dLL_lnv)

	d2lnv_gamma_rho		=	rp.dd_func_lags(self,ll,GARK,	ll.dlnv_e_rho_G,					ll.dLL_lnv)
	d2lnv_gamma_lambda	=	rp.dd_func_lags(self,ll,GARK, 	ll.dlnv_e_lambda_G,					ll.dLL_lnv)
	d2lnv_gamma_beta	=	rp.dd_func_lags(self,ll,GARK, 	ll.dlnv_e_beta_G,					ll.dLL_lnv)
	d2lnv_gamma_z		=	rp.dd_func_lags(self,ll,GARK, 	ll.dlnv_z_G,						ll.dLL_lnv)

	d2lnv_psi_rho		=	rp.dd_func_lags(self,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv)
	d2lnv_psi_lambda	=	rp.dd_func_lags(self,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv)
	d2lnv_psi_beta		=	rp.dd_func_lags(self,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv)
	d2lnv_psi_z			=	rp.dd_func_lags(self,ll,GARM, 	ll.h_z_val,							ll.dLL_lnv)
	GARM=0#Releases memory
	GARK=0#Releases memory

	AMAq=-rp.ARMA_product(ll.AMA_1,self.L,self.q)
	d2lnv_lambda2,		d2e_lambda2		=	rp.dd_func_lags_mult(self,ll,AMAq,	ll.de_lambda,	ll.de_lambda,	'lambda',	'lambda', transpose=True)
	d2lnv_lambda_rho,	d2e_lambda_rho	=	rp.dd_func_lags_mult(self,ll,AMAq,	ll.de_lambda,	ll.de_rho,		'lambda',	'rho' )
	d2lnv_lambda_beta,	d2e_lambda_beta	=	rp.dd_func_lags_mult(self,ll,AMAq,	ll.de_lambda,	ll.de_beta,		'lambda',	'beta')
	AMAq=0#Releases memory


	AMAp=-rp.ARMA_product(ll.AMA_1,self.L,self.p)
	d2lnv_rho_beta,		d2e_rho_beta	=	rp.dd_func_lags_mult(self,ll,AMAp,	ll.de_rho,		ll.de_beta,		'rho',		'beta', de_zeta_u=-self.X)


	d2lnv_mu_rho			=	rp.dd_func_lags(self,ll,None, 		rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=1) 
	d2lnv_mu_lambda			=	rp.dd_func_lags(self,ll,None, 		rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv, 	addavg=1) 
	d2lnv_mu_beta			=	rp.dd_func_lags(self,ll,None, 		rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=1) 
	d2lnv_mu_z				=	rp.dd_func_lags(self,ll,None, 		ll.h_z_val,							ll.dLL_lnv, 	addavg=1) 

	d2lnv_z2				=	rp.dd_func_lags(self,ll,ll.GAR_1MA, ll.h_2z_val,						ll.dLL_lnv, 	addavg=ll.args_d['mu']) 
	d2lnv_z_rho				=	rp.dd_func_lags(self,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=ll.args_d['mu']) 
	d2lnv_z_lambda			=	rp.dd_func_lags(self,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_lambda)),ll.dLL_lnv, 	addavg=ll.args_d['mu']) 
	d2lnv_z_beta			=	rp.dd_func_lags(self,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=ll.args_d['mu']) 

	d2lnv_rho2,	d2e_rho2	=	rp.dd_func_lags_mult(self,ll,	None,	ll.de_rho,		ll.de_rho,		'rho',		'rho' )
	AMAp=0#Releases memory

	d2lnv_beta2,d2e_beta2	=	rp.dd_func_lags_mult(self,ll,	None,	ll.de_beta,		ll.de_beta,		'beta',		'beta')


	#return debug.hessian_debug(self,args):
	d=self.second_derivatives(ll)
	#Final:
	
	d2LL_de2=-ll.v_inv*self.included
	d2LL_dln_de=ll.e_RE*ll.v_inv*self.included
	d2LL_dln2=-0.5*ll.e_REsq*ll.v_inv*self.included	
	(de_rho_RE,de_lambda_RE,de_beta_RE)=(ll.de_rho_RE,ll.de_lambda_RE,ll.de_beta_RE)
	(dlnv_e_rho,dlnv_e_lambda,dlnv_e_beta)=(ll.dlnv_e_rho,ll.dlnv_e_lambda,ll.dlnv_e_beta)
	(dlnv_gamma,dlnv_psi)=(ll.dlnv_gamma,ll.dlnv_psi)
	(dlnv_mu, dlnv_z)=(ll.dlnv_mu, ll.dlnv_z)			

	D2LL_beta2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dlnv_e_beta, 	dlnv_e_beta,	d2e_beta2, 					d2lnv_beta2)
	D2LL_beta_rho		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dlnv_e_beta, 	dlnv_e_rho,		T(d2e_rho_beta), 		T(d2lnv_rho_beta))
	D2LL_beta_lambda	=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dlnv_e_beta, 	dlnv_e_lambda,	T(d2e_lambda_beta), 	T(d2lnv_lambda_beta))
	D2LL_beta_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_beta))
	D2LL_beta_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_beta))
	D2LL_beta_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	self.W_a,		None, 					None)
	D2LL_beta_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_mu,		None, 					T(d2lnv_mu_beta))
	D2LL_beta_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_z,			None, 					T(d2lnv_z_beta))
	
	D2LL_rho2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dlnv_e_rho, 	dlnv_e_rho,		d2e_rho2, 					d2lnv_rho2)
	D2LL_rho_lambda		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dlnv_e_rho, 	dlnv_e_lambda,	T(d2e_lambda_rho), 		T(d2lnv_lambda_rho))
	D2LL_rho_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_rho))	
	D2LL_rho_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_rho))
	D2LL_rho_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	self.W_a,		None, 					None)
	D2LL_rho_mu			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_mu,		None, 					T(d2lnv_mu_rho))
	D2LL_rho_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_z,			None, 					T(d2lnv_z_rho))
	
	D2LL_lambda2		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dlnv_e_lambda, 	dlnv_e_lambda,	T(d2e_lambda2), 		T(d2lnv_lambda2))
	D2LL_lambda_gamma	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_lambda))
	D2LL_lambda_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_lambda))
	D2LL_lambda_omega	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	self.W_a,		None, 					None)
	D2LL_lambda_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_mu,		None, 					T(d2lnv_mu_lambda))
	D2LL_lambda_z		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_z,			None, 					T(d2lnv_z_lambda))
	
	D2LL_gamma2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma2))
	D2LL_gamma_psi		=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_gamma, 	ll.dlnv_psi,		None, 					d2lnv_gamma_psi)
	D2LL_gamma_omega	=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	self.W_a,		None, 					None)
	D2LL_gamma_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_mu,		None, 					None)
	D2LL_gamma_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_z,			None, 					d2lnv_gamma_z)
	
	D2LL_psi2			=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_psi, 		ll.dlnv_psi,		None, 					None)
	D2LL_psi_omega		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		self.W_a,		None, 					None)
	D2LL_psi_mu			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_mu,		None, 					None)
	D2LL_psi_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_z,			None, 					d2lnv_psi_z)
	
	D2LL_omega2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			self.W_a, 		self.W_a,		None, 					None)
	D2LL_omega_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			self.W_a, 		dlnv_mu,		None, 					None)
	D2LL_omega_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			self.W_a, 		dlnv_z,			None, 					None)
	
	D2LL_mu2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_mu,		None, 					None)
	D2LL_mu_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_z,			None, 					d2lnv_mu_z)
	
	D2LL_z2				=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_z, 		dlnv_z,			None, 					d2lnv_z2)
	
	
	H= [[D2LL_beta2,			D2LL_beta_rho,		D2LL_beta_lambda,		D2LL_beta_gamma,	D2LL_beta_psi,		D2LL_beta_omega,	D2LL_beta_mu,	D2LL_beta_z		],
		[T(D2LL_beta_rho),		D2LL_rho2,			D2LL_rho_lambda,		D2LL_rho_gamma,		D2LL_rho_psi,		D2LL_rho_omega,		D2LL_rho_mu,	D2LL_rho_z			],
		[T(D2LL_beta_lambda),	T(D2LL_rho_lambda),	D2LL_lambda2,			D2LL_lambda_gamma,	D2LL_lambda_psi,	D2LL_lambda_omega,	D2LL_lambda_mu,	D2LL_lambda_z		],
		[T(D2LL_beta_gamma),	T(D2LL_rho_gamma),	T(D2LL_lambda_gamma),	D2LL_gamma2,		D2LL_gamma_psi,		D2LL_gamma_omega, 	D2LL_gamma_mu,	D2LL_gamma_z		],
		[T(D2LL_beta_psi),		T(D2LL_rho_psi),	T(D2LL_lambda_psi),		T(D2LL_gamma_psi),	D2LL_psi2,			D2LL_psi_omega, 	D2LL_psi_mu,	D2LL_psi_z			],
		[T(D2LL_beta_omega),	T(D2LL_rho_omega),	T(D2LL_lambda_omega),	T(D2LL_gamma_omega),T(D2LL_psi_omega),	D2LL_omega2, 		D2LL_omega_mu,	D2LL_omega_z		], 
		[T(D2LL_beta_mu),		T(D2LL_rho_mu),		T(D2LL_lambda_mu),		T(D2LL_gamma_mu),	T(D2LL_psi_mu),		T(D2LL_omega_mu), 	D2LL_mu2,		D2LL_mu_z			],
		[T(D2LL_beta_z),		T(D2LL_rho_z),		T(D2LL_lambda_z),		T(D2LL_gamma_z),	T(D2LL_psi_z),		T(D2LL_omega_z), 	D2LL_mu_z,		D2LL_z2				]]


	H=rp.concat_matrix(H)
	#Hn=-debug.hess_debug(self,ll,0.000001)#debugging
	#debug.LL_debug_detail(self,ll,0.0000001)

	return H 


	

def dd_func_lags_mult(panel,AMAL,de_xi,de_zeta,vname1,vname2,transpose=False, de_zeta_u=None,d=0.000001):
	#de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"

	if de_xi is None or de_zeta is None:
		return None,None	
	(N,T,m)=de_xi.shape
	(N,T,k)=de_zeta.shape
	DLL_e=panel.DLL_e.reshape(N,T,1,1)
	u_calc=False
	if de_zeta_u is None:
		de_zeta_u=de_zeta#for error beta-rho covariance, the u derivative must be used
	#ARIMA:
	#*********************1
	if not AMAL is None:
		de2_zeta_xi_1=fu.dot(AMAL,de_zeta_u)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi_1=de2_zeta_xi_1+np.swapaxes(de2_zeta_xi_1,2,3)#adds the transpose
		de2_zeta_xi_RE_1=rp.ddRE(panel,de2_zeta_xi_1,de_xi,de_zeta,panel.e,vname1,vname2)
	else:
		de2_zeta_xi_1=0
		de2_zeta_xi_RE_1=rp.ddRE(panel,None,de_xi,de_zeta,panel.e,vname1,vname2)
		if de2_zeta_xi_RE_1 is None:
			de2_zeta_xi_RE_1=None
	if not de2_zeta_xi_RE_1 is None:	
		de2_zeta_xi_RE_1 = de2_zeta_xi_RE_1 * DLL_e
		de2_zeta_xi_RE_1 = np.sum(np.sum(de2_zeta_xi_RE_1,0),0)#and sum it
	#*********************2
	if not AMAL is None:
		de2_zeta_xi_2=fu.dot(AMAL,de_zeta_u)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi_2=de2_zeta_xi_2+np.swapaxes(de2_zeta_xi_2,2,3)#adds the transpose
		de2_zeta_xi_RE_2=rp.ddRE(panel,de2_zeta_xi_2,de_xi,de_zeta,panel.e,vname1,vname2)
	else:
		de2_zeta_xi_2=0
		de2_zeta_xi_RE_2=rp.ddRE(panel,None,de_xi,de_zeta,panel.e,vname1,vname2)
		if de2_zeta_xi_RE_2 is None:
			de2_zeta_xi_RE_2=None
	if not de2_zeta_xi_RE_2 is None:	
		de2_zeta_xi_RE_2 = de2_zeta_xi_RE_2 * DLL_e
		de2_zeta_xi_RE_2 = np.sum(np.sum(de2_zeta_xi_RE_2,0),0)#and sum it
	#*********************3
	if not AMAL is None:
		de2_zeta_xi_3=fu.dot(AMAL,de_zeta_u)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi_3=de2_zeta_xi_3+np.swapaxes(de2_zeta_xi_3,2,3)#adds the transpose
		de2_zeta_xi_RE_3=rp.ddRE(panel,de2_zeta_xi_3,de_xi,de_zeta,panel.e,vname1,vname2)
	else:
		de2_zeta_xi_3=0
		de2_zeta_xi_RE_3=rp.ddRE(panel,None,de_xi,de_zeta,panel.e,vname1,vname2)
		if de2_zeta_xi_RE_3 is None:
			de2_zeta_xi_RE_3=None
	if not de2_zeta_xi_RE_3 is None:	
		de2_zeta_xi_RE_3 = de2_zeta_xi_RE_3 * DLL_e
		de2_zeta_xi_RE_3 = np.sum(np.sum(de2_zeta_xi_RE_3,0),0)#and sum it
	#*********************4
	if not AMAL is None:
		de2_zeta_xi_4=fu.dot(AMAL,de_zeta_u)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi_4=de2_zeta_xi_4+np.swapaxes(de2_zeta_xi_4,2,3)#adds the transpose
		de2_zeta_xi_RE_4=rp.ddRE(panel,de2_zeta_xi_4,de_xi,de_zeta,panel.e,vname1,vname2)
	else:
		de2_zeta_xi_4=0
		de2_zeta_xi_RE_4=rp.ddRE(panel,None,de_xi,de_zeta,panel.e,vname1,vname2)
		if de2_zeta_xi_RE_4 is None:
			de2_zeta_xi_RE_4=None
	if not de2_zeta_xi_RE_4 is None:	
		de2_zeta_xi_RE_4 = de2_zeta_xi_RE_4 * DLL_e
		de2_zeta_xi_RE_4 = np.sum(np.sum(de2_zeta_xi_RE_4,0),0)#and sum it
	#GARCH: 
	if panel.m>0:
		h_e_de2_zeta_xi = de2_zeta_xi * panel.h_e_val.reshape(N,T,1,1) * DLL_e
		h_2e_dezeta_dexi = panel.h_2e_val.reshape(N,T,1,1) * de_xi.reshape((N,T,m,1)) * de_zeta.reshape((N,T,1,k))

		d2lnv_zeta_xi = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)

		d_mu = panel.c[panel.mu_sel] * (np.sum(d2lnv_zeta_xi,1) / panel.T_arr.reshape((N,1,1)))
		d_mu = d_mu.reshape((N,1,m,k)) * panel.a.reshape((N,T,1,1))	

		d2lnv_zeta_xi = d2lnv_zeta_xi + d_mu
		d2lnv_zeta_xi = fu.dot(panel.GAR_1MA, d2lnv_zeta_xi)

		d2lnv_zeta_xi=np.sum(np.sum(d2lnv_zeta_xi,0),0)
	else:
		d2lnv_zeta_xi=None

	return d2lnv_zeta_xi,de2_zeta_xi_RE

def dd_func_lags(panel,L,d,dLL,addavg=0, transpose=False):
	#d is "N x T x m" and L is "k x T x T"
	if panel.m==0:
		return None
	if d is None:
		return None		
	(N,T,m)=d.shape
	if L is None:
		x=0
		k=1
	elif len(L)==0:
		return None
	else:
		(k,T,T)=L.shape
		x=np.dot(L,d)#this creates a "k x T x N x m" matrix
		x=np.swapaxes(x,0,2)#changes "k x T x N x m" to "T x N x k x m"
		if transpose:#only happens if lags==k
			x=x+np.swapaxes(x,2,3)#adds the transpose
	if addavg:
		addavg=(addavg*np.sum(d,1)/panel.T_arr).reshape(N,1,k,m)
		x=x+addavg
	dLL=dLL.reshape((N,T,1,1))
	return np.sum(np.sum(dLL*x,1),0)#and sum it	



def savevar(variable,name='tmp'):
	"""takes var and name and saves var with filname <name>.csv """	
	savevars(((variable,name),))

def savevars(varlist):
	"""takes a tuple of (var,name) pairs and saves numpy array var 
	with <name>.csv. Use double brackets for single variable."""	
	if not os.path.exists('/output'):
		os.makedirs('/output')	
	for var,name in varlist:
		name=name.replace('.csv','')
		np.savetxt("%s\\output\\%s.csv" %(os.getcwd(),name),var,delimiter=";")


def T(x):
	if x is None:
		return None
	return x.T



