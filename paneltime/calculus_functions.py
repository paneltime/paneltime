#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import stat_functions as stat
from scipy import sparse as sp
import debug
try:
	import cfunctions as c
except:
	c=None


def dd_func_lags_mult(panel,ll,g,AMAL,vname1,vname2,transpose=False, u_gradient=False):
	de2_zeta_xi_RE,de2_zeta_xi 	= dd_func_lags_mult_arima(panel,ll,g,AMAL,vname1,vname2,transpose, u_gradient)
	dd_re_variance 							= dd_func_re_variance(panel,ll,g,vname1,vname2,de2_zeta_xi_RE,u_gradient)
	d2LL_d2lnv_zeta_xi,d2LL_d2e_zeta_xi_RE 	= dd_func_garch(panel,ll,g,vname1,vname2,de2_zeta_xi_RE,de2_zeta_xi,dd_re_variance,u_gradient)
	
	return d2LL_d2lnv_zeta_xi,d2LL_d2e_zeta_xi_RE

def dd_func_lags_mult_arima(panel,ll,g,AMAL,vname1,vname2,transpose, u_gradient):
	#de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"
	de_xi=g.__dict__['de_'+vname1]
	de_zeta=g.__dict__['de_'+vname2]
	de2_zeta_xi_RE=None
	if de_xi is None or de_zeta is None:
		return None,None
	(N,T,m)=de_xi.shape
	(N,T,k)=de_zeta.shape	
	#ARIMA:
	if not AMAL is None:
		if u_gradient:
			de2_zeta_xi=-panel.arma_dot.dot(AMAL,panel.XIV,ll)#"T x N x s x m #for error beta-rho covariance, the u gradient must be used	
		else:
			de2_zeta_xi=panel.arma_dot.dot(AMAL,de_zeta,ll)#"T x N x s x m
		if transpose:#only happens if lags==k
			de2_zeta_xi=de2_zeta_xi+np.swapaxes(de2_zeta_xi,2,3)#adds the transpose
		de2_zeta_xi=de2_zeta_xi*panel.included[4]
	else:
		de2_zeta_xi=None
	de2_zeta_xi_RE=add(([de2_zeta_xi, ll.re_obj_i.ddRE(de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2,panel), ll.re_obj_t.ddRE(de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2,panel)]),True)
	
	return de2_zeta_xi_RE,de2_zeta_xi


def dd_func_re_variance(panel,ll,g,vname1,vname2,de2_zeta_xi_RE,u_gradient):
	if panel.N<=1 or panel.options.fixed_random_group_eff.value==0:
		return None
	#voaltility RE:

	de_xi_RE=g.__dict__['de_'+vname1+'_RE']
	de_zeta_RE=g.__dict__['de_'+vname2+'_RE']
	if de_xi_RE is None or de_zeta_RE is None:
		return None
	(N,T,m)=de_xi_RE.shape
	(N,T,k)=de_zeta_RE.shape		
	de_xi_RE_r=de_xi_RE.reshape((N,T,m,1))
	de_zeta_RE_r=de_zeta_RE.reshape((N,T,1,k))	
	incl=panel.included[4]
	
	dvRE_xi=g.__dict__['dvRE_'+vname1].reshape((N,T,m,1))
	dvRE_zeta=g.__dict__['dvRE_'+vname2].reshape((N,T,1,k))			
	
	if not de2_zeta_xi_RE is None:
		d_xi_input=g.__dict__['d_'+vname1+'_input']
		d_zeta_input=g.__dict__['d_'+vname2+'_input']		
		try:
			dd_e_RE_sq=(2*de_xi_RE_r*de_zeta_RE_r+2*ll.e_RE.reshape(N,T,1,1)*de2_zeta_xi_RE)*incl
		except:
			dd_e_RE_sq=(2*de_xi_RE_r*de_zeta_RE_r+2*ll.e_RE.reshape(N,T,1,1)*de2_zeta_xi_RE)*incl
		ddmeane2= panel.mean(dd_e_RE_sq,(0,1))*incl
		dd_input=(dd_e_RE_sq-ddmeane2)*incl

		ddvRE_d_xi_zeta=ddmeane2-add((ll.re_obj_i_v.ddRE(dd_input,d_xi_input,d_zeta_input,ll.varRE_input,vname1,vname2,panel),
									  ll.re_obj_t_v.ddRE(dd_input,d_xi_input,d_zeta_input,ll.varRE_input,vname1,vname2,panel)),True)
	else:
		ddvRE_d_xi_zeta=None
		
	dlnvRE,		ddlnvRE		=	ll.dlnvRE.reshape((N,T,1,1)),		ll.ddlnvRE.reshape((N,T,1,1))
	dd_re_variance	=	add((prod((dlnvRE,ddvRE_d_xi_zeta))	,	ddlnvRE*dvRE_xi*dvRE_zeta))
	return dd_re_variance


def dd_func_garch(panel,ll,g,vname1,vname2,de2_zeta_xi_RE,de2_zeta_xi,dd_re_variance,u_gradient):
	#GARCH: 
	incl=panel.included[4]
	if (g.__dict__['de_'+vname1] is None) or (g.__dict__['de_'+vname2] is None):
		return None, None	
	(N,T,m)=g.__dict__['de_'+vname1].shape
	(N,T,k)=g.__dict__['de_'+vname2].shape
	DLL_e=g.DLL_e.reshape(N,T,1,1)
	
	d2LL_d2e_zeta_xi_RE=None
	if not de2_zeta_xi_RE is None:	
		d2LL_d2e_zeta_xi_RE = de2_zeta_xi_RE * DLL_e	
		d2LL_d2e_zeta_xi_RE = np.sum(np.sum(d2LL_d2e_zeta_xi_RE*incl,0),0)
		
	RE_suffix='_RE'
	if (not panel.options.RE_in_GARCH.value):
		de2_zeta_xi_RE=de2_zeta_xi
		RE_suffix=''
	de_xi_RE=g.__dict__['de_'+vname1+RE_suffix]
	de_zeta_RE=g.__dict__['de_'+vname2+RE_suffix]	

	
	dLL_lnv=g.dLL_lnv.reshape(N,T,1,1)
	if de2_zeta_xi_RE is None:
		de2_zeta_xi_RE=0
	d2lnv_zeta_xi=None
	d2LL_d2lnv_zeta_xi=None
	d_omega_e=None
	d2lnv_zeta_xi_h=None
	if panel.pqdkm[4]>0:
		if u_gradient:
			de_zeta_RE=g.__dict__['de_'+vname2+RE_suffix]
		h_e_de2_zeta_xi =  ll.h_e_val.reshape(N,T,1,1)  * de2_zeta_xi_RE
		h_2e_dezeta_dexi = ll.h_2e_val.reshape(N,T,1,1) * de_xi_RE.reshape((N,T,m,1)) * de_zeta_RE.reshape((N,T,1,k))

		d2lnv_zeta_xi_h = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)
		
		d2lnv_zeta_xi_h = panel.arma_dot.dot(ll.GAR_1MA, d2lnv_zeta_xi_h,ll)
		
	d2lnv_zeta_xi = add((d2lnv_zeta_xi_h,  dd_re_variance,d_omega_e), True)
	if not d2lnv_zeta_xi is None:
		d2LL_d2lnv_zeta_xi = np.sum(prod((d2lnv_zeta_xi,dLL_lnv*incl),True),(0,1))
	
	return d2LL_d2lnv_zeta_xi,d2LL_d2e_zeta_xi_RE

def dd_func_lags(panel,ll,L,d,dLL,transpose=False):
	#d is "N x T x m" and L is "k x T x T"
	if panel.pqdkm[4]==0:
		return None
	if d is None:
		return None		
	(N,T,m)=d.shape
	if L is None:
		x=0
	elif len(L)==0:
		return None
	elif type(L)==tuple:#elif len(L.shape)==3:
		x=panel.arma_dot.dot(L,d,ll)#"T x N x k x m"
		if x is None:
			return None
	elif len(L.shape)==2:
		x=dot(L,d).reshape(N,T,1,m)
	dLL=dLL.reshape((N,T,1,1))
	return np.sum(np.sum(dLL*x,1),0)#and sum it	

def dd_omega_func(dlnv_omega_e,de_xi):
	if de_xi is None:
		return None
	N,T,k=de_xi.shape
	N,T,m=dlnv_omega_e.shape	
	return np.sum(dlnv_omega_e*de_xi,(0,1)).reshape((k,m))

def add(iterable,ignore=False):
	"""Sums iterable. If ignore=True all elements except those that are None are added. If ignore=False, None is returned if any element is None. """
	x=None
	for i in range(len(iterable)):
		if not iterable[i] is None:
			if x is None:
				x=iterable[i]
			else:
				x=x+iterable[i]
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

def sumNT(nparray):
	if nparray is None:
		return None
	s=nparray.shape
	if len(s)<3:
		raise RuntimeError("Not enough dimensions")
	
	return np.sum(nparray.reshape(list(s)+[1]),(0,1))
	

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


def ARMA_product(m,k,panel,ll,letter):
	return panel.arma_dot.roll_array(m,k,letter,ll)
	a=[]
	for i in range(k):
		a.append(roll(panel.arma_dot.conv(m,ll),-i-1,1))
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
	ret[tuple(v)]=empty_val

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
	
	
class arma_dot_obj:
	def __init__(self,n,pqdkm):
		a=np.arange(n)
		A=np.tril(np.array([np.roll(a,i) for i in range(n)]).T)
		self.nz=np.nonzero(A+np.diag(np.ones(n)))
		self.az=A[self.nz]
		self.mdict={}
		for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
			self.mdict[i]=np.zeros((n,n))
			
		self.rdict={}
		p,q,d,k,m=pqdkm
		for name, i in [('AMA_1_p',p),('AMA_1_q',q),('GAR_1_k',k),('GAR_1_m',m)]:
			self.rdict[name]=np.zeros((i,n,n))
		
		
	def dotroll(self,aband,k,sign,b,ll):
		
		
		ARMA_m=self.conv(aband,ll)
		s=list(b.shape)
		x=np.moveaxis(b,1,0)
		s2=x.shape
		x.resize((s[1],s[0]*np.prod(s[2:])))
		x=sign*np.dot(ARMA_m,x)
		x.resize(s2)
		x=np.moveaxis(x,1,0)
		w=[]
		for i in range(k):
			w.append(np.roll(np.array(x),i+1,1))
			w[i][:,:i+1]=0
		x=np.array(w)
		x=np.moveaxis(x,0,2)	
		return x
			
	def dot(self,a,b,ll):
		if len(a)>2:
			(M,aband,k,sgn)=a
			if len(M)==0:
				return None
			return self.dotroll(aband, k, sgn, b, ll)
		s=list(b.shape)
		x=np.moveaxis(b,1,0)
		x=x.reshape((s[1],s[0]*np.prod(s[2:])))
		ARMA_m=self.conv(a,ll)
		x=np.dot(ARMA_m,x)
		x.resize([s[1],s[0]]+s[2:])
		x=np.moveaxis(x,0,1)
		return x
	
	def conv(self,ARMA,ll):
		#return ARMA
		ARMA_array,name=ARMA
		if ll.AMA_dict[name] is None or True:
			self.mdict[name][self.nz]=ARMA_array[self.az]
			ll.AMA_dict[name]=self.mdict[name]
		return ll.AMA_dict[name]
	
	def roll_array(self,ARMA,k,letter,ll):
		ARMA_array,name=ARMA
		M=self.conv(ARMA,ll)
		for i in range(k):
			self.rdict[f"{name}_{letter}"][i]=roll(M,-i-1,1)
		return self.rdict[f"{name}_{letter}"]



def test(a,b):
	return a+b