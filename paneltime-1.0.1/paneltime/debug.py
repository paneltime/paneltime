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
		if panel.N>1:
			lnv_0=lnv_0+args['mu']*avg_h_0	
		lnv_0=np.maximum(np.minimum(lnv_0,709),-709)
	v_0=np.exp(lnv_0)*panel.a
	v_inv_0=np.exp(-lnv_0)*panel.a	
	e_RE_0=rp.RE(ll,panel,e_0)
	e_REsq_0=e_RE_0**2
	LL_value_0=ll.LL_const-0.5*np.sum((lnv_0+(e_REsq_0)*v_inv_0)*panel.included)	

	#****LL1*****
	matrices=set_garch_arch_debug(panel,args,0)
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
		if panel.N>1:
			lnv_1=lnv_1+(args['mu']+d)*avg_h_1	
		lnv_1=np.maximum(np.minimum(lnv_1,709),-709)
	v_1=np.exp(lnv_1)*panel.a
	v_inv_1=np.exp(-lnv_1)*panel.a	
	e_RE_1=rp.RE(ll,panel,e_1)
	e_REsq_1=e_RE_1**2
	LL_value_1=ll.LL_const-0.5*np.sum((lnv_1+(e_REsq_1)*v_inv_1)*panel.included)	
	
	#****LL2*****
	matrices=set_garch_arch_debug(panel,args,0)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	beta_d=beta_*0
	beta_d[0][0]=0
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
		if panel.N>1:
			lnv_2=lnv_2+args['mu']*avg_h_2
		lnv_2=np.maximum(np.minimum(lnv_2,709),-709)
	v_2=np.exp(lnv_2)*panel.a
	v_inv_2=np.exp(-lnv_2)*panel.a	
	e_RE_2=rp.RE(ll,panel,e_2)
	e_REsq_2=e_RE_2**2
	LL_value_2=ll.LL_const-0.5*np.sum((lnv_2+(e_REsq_2)*v_inv_2)*panel.included)
	
	#****LL3*****
	matrices=set_garch_arch_debug(panel,args,0)
	beta_,Wbeta_,AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
	beta_d=beta_*0
	beta_d[0][0]=0
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
		if panel.N>1:
			lnv_3=lnv_3+args['mu']*avg_h_3
		lnv_3=np.maximum(np.minimum(lnv_3,709),-709)
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


def set_garch_arch_debug(self,args,d):

	p,q,m,k,nW=self.p,self.q,self.m,self.k,self.nW
	beta,rho,lambda_,gamma,psi,Wbeta=args['beta'],args['rho'],args['lambda'],args['gamma'],args['psi'],args['omega']
	beta_=beta.reshape((len(beta),1))
	Wbeta_=Wbeta.reshape((len(Wbeta),1))
	X=self.I+rp.lag_matr(self.L,self.zero,q,lambda_)
	try:
		AMA_1=np.linalg.inv(X)
	except:
		return None
	AAR=self.I-rp.lag_matr(self.L,self.zero,p,rho+d)
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=self.I-rp.lag_matr(self.L,self.zero,k,gamma)
	try:
		GAR_1=np.linalg.inv(X)
	except:
		return None
	GMA=rp.lag_matr(self.L,self.zero,m,psi)	
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



def T(x):
	if x is None:
		return None
	return x.T



