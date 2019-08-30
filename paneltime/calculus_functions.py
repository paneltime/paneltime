#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import stat_functions as stat
from scipy import sparse as sp



def dd_func_lags_mult(panel,ll,g,AMAL,de_xi,de_zeta,vname1,vname2,transpose=False, de_zeta_u=None):
	#de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"
	
	if de_xi is None or de_zeta is None:
		return None,None	
	(N,T,m)=de_xi.shape
	(N,T,k)=de_zeta.shape
	DLL_e=g.DLL_e.reshape(N,T,1,1)
	u_calc=False
	if de_zeta_u is None:
		de_zeta_u=de_zeta#for error beta-rho covariance, the u derivative must be used
	#ARIMA:
	if not AMAL is None:
		de2_zeta_xi=dot(AMAL,de_zeta_u,False)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi=de2_zeta_xi+np.swapaxes(de2_zeta_xi,2,3)#adds the transpose
		de2_zeta_xi_RE=de2_zeta_xi+ll.re_obj_i.ddRE(de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2)+ll.re_obj_t.ddRE(de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2)
	else:
		de2_zeta_xi=0
		de2_zeta_xi_RE=ll.re_obj_i.ddRE(None,de_xi,de_zeta,ll.e,vname1,vname2)+ll.re_obj_t.ddRE(None,de_xi,de_zeta,ll.e,vname1,vname2)
		if de2_zeta_xi_RE is None:
			de2_zeta_xi_RE=None
	if not de2_zeta_xi_RE is None:	
		de2_zeta_xi_RE = de2_zeta_xi_RE * DLL_e
		de2_zeta_xi_RE = np.sum(np.sum(de2_zeta_xi_RE,0),0)#and sum it

	#GARCH: 
	if panel.m>0:
		de_xi   = de_xi.reshape((N,T,m,1))
		de_zeta = de_zeta.reshape((N,T,1,k))
		h_e_de2_zeta_xi =  ll.h_e_val.reshape(N,T,1,1)  * de2_zeta_xi
		h_2e_dezeta_dexi = ll.h_2e_val.reshape(N,T,1,1) * de_xi * de_zeta

		d2lnv_zeta_xi_h = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)
		
		if panel.N>1:
			if ll.zmu:
				h_e_val,h_2e_val,incl =ll.h_e_val.reshape(N,T,1,1),ll.h_2e_val.reshape(N,T,1,1),panel.included.reshape((N,T,1,1))
				e_de2_zeta_xi   = panel.mean(h_e_val * de2_zeta_xi,1)
				e2_dezeta_dexi  = panel.mean(de_xi*de_zeta*h_2e_val,1)					
			else:
				avg_e2,davg_lne2,incl =ll.avg_e2.reshape(N,1,1,1),ll.davg_lne2.reshape(N,T,1,1),panel.included.reshape((N,T,1,1))	
				e_de2_zeta_xi   = panel.group_var_wght*panel.mean(davg_lne2 * de2_zeta_xi,1)
				e2_dezeta_dexi  = panel.group_var_wght*2*panel.mean(de_xi*de_zeta/avg_e2,1)
				e2_dezeta_dexi -= panel.group_var_wght*panel.mean(davg_lne2*de_xi,1)*panel.mean(davg_lne2*de_zeta,1)

			d2lnv_zeta_xi_e = (e_de2_zeta_xi+e2_dezeta_dexi).reshape(N,1,m,k)
			
			d_mu = ll.args_d['mu'] * d2lnv_zeta_xi_e  * incl

		else:
			d_mu=0
		
		
		d2lnv_zeta_xi_h = dot(ll.GAR_1MA, d2lnv_zeta_xi_h)
		
		d2lnv_zeta_xi = d2lnv_zeta_xi_h + d_mu
		
		d2lnv_zeta_xi=np.sum(np.sum(d2lnv_zeta_xi*g.dLL_lnv.reshape((N,T,1,1)),0),0)
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
		x=dot(L,d,False)#"T x N x k x m"
	elif len(L.shape)==2:
		x=dot(L,d).reshape(N,T,1,m)
	if addavg:#for mu
		addavg=(addavg*panel.mean(d,1)).reshape(N,1,1,m)
		x=x+addavg
	dLL=dLL.reshape((N,T,1,1))
	return np.sum(np.sum(dLL*x,1),0)#and sum it	


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
	if np.any(np.isnan(d0)) or np.any(np.isnan(d1)):
		x=np.empty((k,m))
		x[:]=np.nan
		return x
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



def LU(X):
	"Calculates LU decomposition, where X may be multi dimensional. Not in use, should be tested"
	shape=X.shape
	N=shape[0]
	U=np.zeros((shape))
	L=np.zeros((shape))
	U+=np.diag(np.ones(N))
	X=X*1
	for k in range(N):
		for i in range(k,N):
			s = 0
			for m in range(k):
				s+= L[i,m] * U[m,k]
			L[i,k] = X[i,k] - s
		singularity=(np.abs(L[k,k]) < 1E-18)
		L[k,k]=singularity*1E-18+(1-singularity)*L[k,k]
		for j in range(k+1,N):
			s = 0
			for m in range(k):
				s += L[k,m] * U[m,j]
			U[k,j] = (X[k,j] - s)/ L[k,k]
	return L,U

def minverse(X,L=None,U=None):
	"Calculates the matrix inverse where X may be multi dimensional. Not in use, should be tested"
	shape=X.shape
	N=shape[0]
	if L is None:
		L,U=LU(X)
	I=np.diag(np.ones(N))
	inv=np.zeros(shape)
	tmp=np.zeros(shape[1:])

	for k in range(N):
		tmp[0] = I[0,k] / L[0,0]
		for i in range(1,N):
			Sum = 0
			for j in range(i):
				Sum = Sum + L[i,j] * tmp[j]
			tmp[i] = (I[i,k] - Sum) / L[i,i]
		inv[N-1,k] = tmp[N-1];
		for i in range(N-2,-1,-1):
			Sum = 0
			for j in range(i,N):
				Sum = Sum + U[i,j] * inv[j,k]
			inv[i,k] =tmp[i] - Sum
	return inv	

def diag(X):
	return np.array([X[i,i] for i in range(len(X))])


def mmult(X,Y):
	"""returns the dot product of X*Y """	
	a=X.shape
	b=Y.shape
	out_shape=list(a)
	out_shape[1]=b[1]
	r=np.zeros(out_shape)
	for i in range(a[0]):
		for j in range(b[1]):
			for k in range(a[1]):
				r[i,j]+=X[i,k]*Y[k,j]
	return r

def dots(a):
	a=list(a)
	m=a.pop()
	for i in range(len(a)):
		m=mmult(a.pop(),m)
	return m


def dot(a,b,reduce_dims=True):
	"""Matrix multiplication. Returns the dot product of a*b where either a or be or both to be
	arrays of matrices. Faster than mmult, less general and only used for special purpose.
	Todo: generalize and merge"""
	if type(a)==sp.csc_matrix:
		return a.multiply(b)
	if a is None or b is None:
		return None
	if len(a.shape)==5 and len(b.shape)==5 and a.shape==b.shape:
		return mmult(a,b)
	if len(a.shape)==2 and len(b.shape)==2:
		if a.shape[1]!=b.shape[0] and a.shape[0]==b.shape[0]:
			return np.dot(a.T,b)
		return np.dot(a,b)
	if len(a.shape)==2 and len(b.shape)==3:
		N,T,k=b.shape
		x=np.moveaxis(b, 1, 0)
		x=x.reshape((T,N*k))
		x=np.dot(a,x)
		x.resize((T,N,k))
		x=np.moveaxis(x,0,1)
		#slower alternative:
		#x2=np.array([np.dot(a,b[i]) for i in range(b.shape[0])])
		return x
	elif len(a.shape)==3 and len(b.shape)==2:
		return np.array([np.dot(a[i],b) for i in range(a.shape[0])])
	elif len(a.shape)==3 and len(b.shape)==3:
		if a.shape[1]!=b.shape[1]:
			raise RuntimeError("dimensions do not match")
		elif a.shape[0]==b.shape[0] and reduce_dims:
			x=np.sum([np.dot(a[i].T,b[i]) for i in range(a.shape[0])],0)
			return x
		elif a.shape[2]==b.shape[1]:
			k,Ta,Ta2=a.shape
			if Ta2!=Ta:
				raise RuntimeError("hm")
			N,T,m=b.shape
			b_f=np.moveaxis(b, 1, 0)
			a_f=a.reshape((k*T,T))
			b_f=b_f.reshape((T,N*m))
			x=np.dot(a_f,b_f)
			x.resize((k,T,N,m))	
			x=np.swapaxes(x, 2, 0)
			#slower:
			#x2=np.array([[np.dot(a[i],b[j]) for j in range(b.shape[0])] for i in range(a.shape[0])])
			#x2=np.moveaxis(x2,0,2)	
			return x


	elif len(a.shape)==2 and len(b.shape)==4:
		if a.shape[1]!=b.shape[1] or a.shape[1]!=a.shape[0]:
			raise RuntimeError("dimensions do not match")
		else:
			N,T,k,m=b.shape
			x=np.moveaxis(b, 1, 0)
			x=x.reshape((T,N*k*m))
			x=np.dot(a,x)
			x.resize((T,N,k,m))
			x=np.moveaxis(x,0,1)

			#slower alternatives:
			#x=np.array([[np.dot(a,b[i,:,j]) for i in range(b.shape[0])] for j in range(b.shape[2])])
			#x=np.moveaxis(x,0,2)
				#or
			#x2=np.zeros(b.shape)		
			#r=c.dot(a,b,b.shape,x2)		
			return x

	else:
		raise RuntimeError("this multiplication is not supported by dot")
	
	
