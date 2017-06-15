#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import statproc as stat
import functions as fu



def dd_func_lags_mult(panel,ll,AMAL,de_xi,de_zeta,vname1,vname2,transpose=False, de_zeta_u=None):
	#de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"
	
	if de_xi is None or de_zeta is None:
		return None,None	
	(N,T,m)=de_xi.shape
	(N,T,k)=de_zeta.shape
	DLL_e=ll.DLL_e.reshape(N,T,1,1)
	u_calc=False
	if de_zeta_u is None:
		de_zeta_u=de_zeta#for error beta-rho covariance, the u derivative must be used
	#ARIMA:
	if not AMAL is None:
		de2_zeta_xi=fu.dot(AMAL,de_zeta_u,False)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi=de2_zeta_xi+np.swapaxes(de2_zeta_xi,2,3)#adds the transpose
		de2_zeta_xi_RE=ddRE(ll,panel,de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2)
	else:
		de2_zeta_xi=0
		de2_zeta_xi_RE=ddRE(ll,panel,None,de_xi,de_zeta,ll.e,vname1,vname2)
		if de2_zeta_xi_RE is None:
			de2_zeta_xi_RE=None
	if not de2_zeta_xi_RE is None:	
		de2_zeta_xi_RE = de2_zeta_xi_RE * DLL_e
		de2_zeta_xi_RE = np.sum(np.sum(de2_zeta_xi_RE,0),0)#and sum it

	#GARCH: 
	if panel.m>0:
		h_e_de2_zeta_xi = de2_zeta_xi * ll.h_e_val.reshape(N,T,1,1)
		h_2e_dezeta_dexi = ll.h_2e_val.reshape(N,T,1,1) * de_xi.reshape((N,T,m,1)) * de_zeta.reshape((N,T,1,k))

		d2lnv_zeta_xi = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)
		
		if panel.N>1:
			d_mu = ll.args_d['mu'] * (np.sum(d2lnv_zeta_xi,1) / panel.T_arr.reshape((N,1,1)))
			d_mu = d_mu.reshape((N,1,m,k)) * panel.included.reshape((N,T,1,1))	
		else:
			d_mu=0
		
		
		d2lnv_zeta_xi = fu.dot(ll.GAR_1MA, d2lnv_zeta_xi)
		
		d2lnv_zeta_xi = d2lnv_zeta_xi + d_mu
		
		d2lnv_zeta_xi=np.sum(np.sum(d2lnv_zeta_xi*ll.dLL_lnv.reshape((N,T,1,1)),0),0)
	else:
		d2lnv_zeta_xi=None

	return d2lnv_zeta_xi,de2_zeta_xi_RE

def dd_func_lags(panel,ll,L,d,dLL,addavg=0, transpose=False):
	#d is "N x T x m" and L is "k x T x T"
	if panel.m==0:
		return None
	if d is None:
		return None		
	(N,T,m)=d.shape
	if L is None:
		x=0
	elif len(L)==0:
		return None
	elif len(L.shape)==3:
		x=fu.dot(L,d,False)#"T x N x k x m"
		if transpose:#only happens if lags==k
			x=x+np.swapaxes(x,2,3)#adds the transpose
	elif len(L.shape)==2:
		x=fu.dot(L,d).reshape(N,T,1,m)
	if addavg:
		addavg=(addavg*np.sum(d,1)/panel.T_arr).reshape(N,1,1,m)
		x=x+addavg
	dLL=dLL.reshape((N,T,1,1))
	return np.sum(np.sum(dLL*x,1),0)#and sum it	



def RE(ll,panel,e,recalc=True):
	"""Following Greene(2012) p. 413-414"""
	if panel.FE_RE==0:
		return e
	eFE=FE(panel,e)
	if panel.FE_RE==1:
		return ll.eFE
	if recalc:
		ll.eFE=eFE
		ll.vLSDV=np.sum(ll.eFE**2)/panel.NT_afterloss
		ll.theta=(1-np.sqrt(ll.vLSDV/(panel.grp_v*panel.n_i)))*panel.included
		ll.theta=np.maximum(ll.theta,0)
		if np.any(ll.theta>1):
			raise RuntimeError("WTF")
	eRE=FE(panel,e,ll.theta)
	return eRE

def dRE(ll,panel,de,e,vname):
	"""Returns the first and second derivative of RE"""
	if not hasattr(ll,'deFE'):
		ll.deFE=dict()
		ll.dvLSDV=dict()
		ll.dtheta=dict()

	if panel.FE_RE==0:
		return de
	elif panel.FE_RE==1:
		return FE(panel,de)	
	sqrt_expr=np.sqrt(1/(panel.grp_v*panel.n_i*ll.vLSDV))
	ll.deFE[vname]=FE(panel,de)
	ll.dvLSDV[vname]=np.sum(np.sum(ll.eFE*ll.deFE[vname],0),0)/panel.NT_afterloss
	ll.dtheta[vname]=-sqrt_expr*ll.dvLSDV[vname]*panel.included

	dRE0=FE(panel,de,ll.theta)
	dRE1=FE(panel,e,ll.dtheta[vname],True)
	return (dRE0+dRE1)*panel.included

def ddRE(ll,panel,dde,de1,de2,e,vname1,vname2):
	"""Returns the first and second derivative of RE"""
	if panel.FE_RE==0:
		return dde
	elif panel.FE_RE==1:
		return FE(panel,dde)	
	(N,T,k)=de1.shape
	(N,T,m)=de2.shape
	if dde is None:
		ddeFE=0
		hasdd=False
	else:
		ddeFE=FE(panel,dde)
		hasdd=True
	eFE=ll.eFE.reshape(N,T,1,1)
	ddvLSDV=np.sum(np.sum(eFE*ddeFE+ll.deFE[vname1].reshape(N,T,k,1)*ll.deFE[vname2].reshape(N,T,1,m),0),0)/panel.NT_afterloss

	ddtheta1=(np.sqrt(1/(panel.grp_v*panel.n_i*(ll.vLSDV**2))))*ll.dvLSDV[vname1].reshape(k,1)*ll.dvLSDV[vname2].reshape(1,m)
	ddtheta2=ddtheta1+(-np.sqrt(1/(panel.grp_v*panel.n_i*ll.vLSDV)))*ddvLSDV
	ddtheta=ddtheta2.reshape(N,1,k,m)*panel.included.reshape(N,T,1,1)

	if hasdd:
		dRE00=FE(panel,dde,ll.theta.reshape(N,T,1,1))
	else:
		dRE00=0
	dRE01=FE(panel,de1.reshape(N,T,k,1),ll.dtheta[vname2].reshape(N,T,1,m),True)
	dRE10=FE(panel,de2.reshape(N,T,1,m),ll.dtheta[vname1].reshape(N,T,k,1),True)
	dRE11=FE(panel,e.reshape(N,T,1,1),ddtheta,True)
	return (dRE00+dRE01+dRE10+dRE11)

def FE(panel,e,w=1,d=False):
	"""returns x after fixed effects, and set lost observations to zero"""
	#assumes e is a "N x T x k" matrix
	if e is None:
		return None
	if len(e.shape)==3:
		N,T,k=e.shape
		s=((N,1,k),(N,T,1))
		n_i=panel.n_i
	elif len(e.shape)==4:
		N,T,k,m=e.shape
		s=((N,1,k,m),(N,T,1,1))
		n_i=panel.n_i.reshape((N,1,1,1))
	ec=e*panel.included.reshape(s[1])
	sum_ec=np.sum(ec,1).reshape(s[0])
	sum_ec_all=np.sum(sum_ec,0)	
	dFE=-(w*(sum_ec/n_i-sum_ec_all/panel.NT_afterloss))*panel.included.reshape(s[1])
	if d==False:
		return ec*panel.included.reshape(s[1])+dFE
	else:
		return dFE


def add(iterable,ignore=False):
	"""Sums iterable. If ignore=True all elements except those that are None are added. If ignore=False, None is returned if any element is None. """
	x=None
	for i in iterable:
		if not i is None:
			if x is None:
				x=i
			else:
				x=x+i
		else:
			if not ignore:
				return None
	return x

def prod(iterable,ignore=False):
	"""Takes the product sum of iterable. If ignore=True all elements except those that are None are multiplied. 
	If ignore=False, None is returned if any element is None. """
	x=None
	for i in iterable:
		if not i is None:
			if x is None:
				x=i
			else:
				x=x*i
		else:
			if not ignore:
				return None
	return x

def fillmatr(X,max_T):
	k=len(X[0])
	z=np.zeros((max_T-len(X),k))
	X=np.concatenate((X,z),0)
	return X

def concat_matrix(block_matrix):
	m=[]
	for i in range(len(block_matrix)):
		r=block_matrix[i]
		C=[]
		for j in range(len(r)):
			if not r[j] is None:
				C.append(r[j])
		if len(C):
			m.append(np.concatenate(C,1))
	m=np.concatenate(m,0)
	return m

def concat_marray(matrix_array):
	arr=[]
	for i in matrix_array:
		if not i is None:
			arr.append(i)
	arr=np.concatenate(arr,2)
	return arr




		
def dd_func(d2LL_de2,d2LL_dln_de,d2LL_dln2,de_dh,de_dg,dln_dh,dln_dg,dLL_de2_dh_dg,dLL_dln2_dh_dg):
	a=[]
	a.append(dd_func_mult(de_dh,d2LL_de2,de_dg))

	a.append(dd_func_mult(de_dh,d2LL_dln_de,dln_dg))
	a.append(dd_func_mult(dln_dh,d2LL_dln_de,de_dg))

	a.append(dd_func_mult(dln_dh,d2LL_dln2,dln_dg))

	a.append(dLL_de2_dh_dg)
	a.append(dLL_dln2_dh_dg)
	return add(a,True)

def dd_func_mult(d0,mult,d1):
	#d0 is N x T x k and d1 is N x T x m
	if d0 is None or d1 is None or mult is None:
		return None
	(N,T,k)=d0.shape
	(N,T,m)=d1.shape
	d0=d0*mult
	d0=np.reshape(d0,(N,T,k,1))
	d1=np.reshape(d1,(N,T,1,m))
	try:
		x=np.sum(np.sum(d0*d1,0),0)#->k x m 
	except RuntimeWarning as e:
		if e.args[0]=='overflow encountered in multiply':
			d0=np.minimum(np.maximum(d0,-1e+100),1e+100)
			d1=np.minimum(np.maximum(d1,-1e+100),1e+100)
			x=np.sum(np.sum(d0*d1,0),0)#->k x m 
		else:
			raise RuntimeWarning(e)
	return x


def ARMA_product(m,k):
	a=[]

	for i in range(k):
		a.append(roll(m,-i-1,1))
	return np.array(a)


def sandwich(H,G,lags=3,ret_hessin=False):
	H=H*1
	sel=[i for i in range(len(H))]
	H[sel,sel]=H[sel,sel]+(H[sel,sel]==0)*1e-15
	hessin=np.linalg.inv(-H)
	V=stat.newey_west_wghts(lags,XErr=G)
	hessinV=fu.dot(hessin,V)
	sandw=fu.dot(hessinV,hessin)
	if ret_hessin:
		return sandw,hessin
	return sandw


	
def differenciate(X,diff,has_intercept):
	for i in range(diff):
		X=X-np.roll(X,1,0)
	X=X[diff:]
	if has_intercept:
		X[:,0]=1
	return X


def roll(a,shift,axis=0,empty_val=0):
	"""For shift>0 (shift<0) this function shifts the shift up (down) by deleting the top (bottom)
	shift and replacing the new botom (top) shift with empty_val"""

	if shift==0:
		return a
	if type(a)==list:
		a=np.a(a)
	s=a.shape

	ret=np.roll(a,shift,axis)
	v=[slice(None)]*len(s)
	if shift>0:
		v[axis]=slice(0,shift)
	else:
		n=s[axis]
		v[axis]=slice(n+shift,n)
		ret[n+shift:]=empty_val
	ret[v]=empty_val

	if False:#for debugging
		arr2=a*1
		arr=a*1
		if len(s)==2:
			T,k=s
			fill=np.ones((abs(shift),k),dtype=arr2.dtype)*empty_val		
			if shift<0:
				ret2= np.append(fill,arr2[0:T+shift],0)
			else:
				ret2= np.append(arr2[shift:],fill,0)		
		elif len(s)==3:
			N,T,k=s
			fill=np.ones((N,abs(shift),k),dtype=arr2.dtype)*empty_val
			if shift<0:
				ret2= np.append(fill,arr2[:,0:T+shift],1)
			else:
				ret2= np.append(arr2[:,shift:],fill,1)		
		elif len(s)==1:
			T=s[0]
			fill=np.ones(abs(shift),dtype=arr2.dtype)*empty_val
			if shift<0:
				ret2= np.append(fill,arr2[0:T+shift],0)
			else:
				ret2= np.append(arr2[shift:],fill,0)	

		if not np.all(ret==ret2):
			raise RuntimeError('Check that the calling procedure has specified the "axis" argument')
	return ret

