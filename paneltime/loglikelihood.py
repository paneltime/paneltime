#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object
import sys
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.win-amd64-3.5'))
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.linux-x86_64-3.5'))
from pathlib import Path
import os
import numpy.ctypeslib as npct
import ctypes as ct
try:#only using c function if installed
	p = os.path.join(Path(__file__).parent.absolute(),'cfunctions')
	cfunct = npct.load_library('ctypes.dll',p)
	#from cfunctions import cextension as c
except ImportError as e:
	c=None
	

import numpy as np
import calculus_ll as cll
import calculus_functions as cf
import random_effects as re
import traceback
import model_parser
import time
import stat_dist
import stat_functions


CDPT = ct.POINTER(ct.c_double) 
CIPT = ct.POINTER(ct.c_uint) 

	

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
	that store dynamic variables that depend on the \n
	If args is a dictionary, the ARMA-GARCH orders are 
	determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
	with the  panel object
	"""
	def __init__(self,args,panel,constraints=None,print_err=False):
		self.err_msg=''
		self.errmsg_h=''

		#checking settings. If the FE/RE is done on the data before LL
		gfre=panel.options.fixed_random_group_eff.value
		tfre=panel.options.fixed_random_time_eff.value
		vfre=panel.options.fixed_random_variance_eff.value
		
		self.re_obj_i=re.re_obj(panel,True,panel.T_i,panel.T_i,gfre)
		self.re_obj_t=re.re_obj(panel,False,panel.date_count_mtrx,panel.date_count,tfre)
		self.re_obj_i_v=re.re_obj(panel,True,panel.T_i,panel.T_i,gfre*vfre)
		self.re_obj_t_v=re.re_obj(panel,False,panel.date_count_mtrx,panel.date_count,tfre*vfre)
		
		self.args=panel.args.create_args(args,panel,constraints)
		self.h_err=""
		self.LL=None
		#self.LL=self.LL_calc(panel)
		try:
			self.LL=self.LL_calc(panel)
			if np.isnan(self.LL):
				self.LL=None						
		except Exception as e:
			if print_err:
				traceback.print_exc()
				print(self.errmsg_h)
		
		


	def LL_calc(self,panel):
		X=panel.XIV

		
		#Idea for IV: calculate Z*u throughout. Mazimize total sum of LL. 
		u = panel.Y-cf.dot(X,self.args.args_d['beta'])
		u_RE = (u+self.re_obj_i.RE(u, panel)+self.re_obj_t.RE(u, panel))*panel.included[3]
		
		matrices=self.arma_calc(panel, u_RE)
		if matrices is None:
			return None		
		AMA_1,AMA_1AR,GAR_1,GAR_1MA, e_RE2, lnv2=matrices		
		e_RE = e_RE2#panel.arma_dot.dot(AMA_1AR,u_RE,self)
		self.h_val, self.h_e_val, self.h_2e_val = e_RE**2*panel.included[3], 2*e_RE*panel.included[3], 2*panel.included[3]
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None	#possibility of additional parameter in sq res function	
		lnv_ARMA = lnv2#panel.arma_dot.dot(GAR_1MA,self.h_val,self)
		
		
		e_REsq =(e_RE**2+(e_RE==0)*1e-18) 
		W_omega = cf.dot(panel.W_a, self.args.args_d['omega'])
		
		
		lnv = W_omega+lnv_ARMA

		LL_full,v,v_inv,self.dlnv_pos=cll.LL(panel,lnv,e_REsq, e_RE)
		self.tobit(panel,LL_full)
		LL=np.sum(LL_full*panel.included[3])
		self.LL_all=np.sum(LL_full)
		self.add_variables(panel,matrices, u, u_RE, lnv_ARMA, lnv, v, W_omega,e_RE,e_REsq,v_inv,LL_full)
		if abs(LL)>1e+100: 
			return None				
		return LL
		
	def add_variables(self,panel,matrices,u, u_RE,lnv_ARMA,lnv,v,W_omega,e_RE,e_REsq,v_inv,LL_full):
		self.v_inv05=v_inv**0.5
		self.e_norm=e_RE*self.v_inv05	
		self.e_norm_centered=(self.e_norm-panel.mean(self.e_norm))*panel.included[3]
		self.u, self.u_RE, self.lnv_ARMA        = u,  u_RE,   lnv_ARMA
		self.lnv,  self.v,    self.LL_full = lnv,       v,    LL_full
		self.W_omega=W_omega
		self.e_RE=e_RE
		self.e_REsq=e_REsq
		self.v_inv=v_inv


	
	def tobit(self,panel,LL):
		if sum(panel.tobit_active)==0:
			return
		g=[1,-1]
		self.F=[None,None]	
		for i in [0,1]:
			if panel.tobit_active[i]:
				I=panel.tobit_I[i]
				self.F[i]= stat_dist.norm(g[i]*self.e_norm[I])
				LL[I]=np.log(self.F[i])
	
	
	def variance_RE(self,panel,e_REsq):
		"""Calculates random/fixed effects for variance."""
		#not in use, expermental. Should be applied to normalize before ARIMA/GARCH
		self.vRE,self.lnvRE,self.dlnvRE=panel.zeros[3],panel.zeros[3],panel.zeros[3]
		self.ddlnvRE,self.dlnvRE_mu,self.ddlnvRE_mu_vRE=panel.zeros[3],None,None
		self.varRE_input, self.ddvarRE_input, self.dvarRE_input = None, None, None
		if panel.options.fixed_random_variance_eff.value==0:
			return panel.zeros[3]
		if panel.N==0:
			return None

		meane2=panel.mean(e_REsq)
		self.varRE_input=(e_REsq-meane2)*panel.included[3]

		mine2=0
		mu=panel.options.variance_RE_norm.value
		self.vRE_i=self.re_obj_i_v.RE(self.varRE_input, panel)
		self.vRE_t=self.re_obj_t_v.RE(self.varRE_input, panel)
		self.meane2=meane2
		vRE=meane2*panel.included[3]-self.vRE_i-self.vRE_t
		self.vRE=vRE
		small=vRE<=mine2
		big=small==False
		vREbig=vRE[big]
		vREsmall=vRE[small]

		lnvREbig=np.log(vREbig+mu)
		lnvREsmall=(np.log(mine2+mu)+((vREsmall-mine2)/(mine2+mu)))
		lnvRE,dlnvRE,ddlnvRE=np.zeros(vRE.shape),np.zeros(vRE.shape),np.zeros(vRE.shape)
		
		lnvRE[big]=lnvREbig
		lnvRE[small]=lnvREsmall
		self.lnvRE=lnvRE*panel.included[3]

		dlnvRE[big]=1/(vREbig+mu)
		dlnvRE[small]=1/(mine2+mu)
		self.dlnvRE=dlnvRE*panel.included[3]
		
		ddlnvRE[big]=-1/(vREbig+mu)**2
		self.ddlnvRE=ddlnvRE*panel.included[3]
	
		return self.lnvRE
		


	def standardize(self,panel,reverse_difference=False):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self. 
		If reverse_difference and the ARIMA difference term d>0, the standardized variables are converted to
		the original undifferenced order. This may be usefull if the predicted values should be used in another 
		differenced regression."""
		if hasattr(self,'Y_st'):
			return		
		m=panel.lost_obs
		N,T,k=panel.X.shape
		if model_parser.DEFAULT_INTERCEPT_NAME in panel.args.names_d['beta']:
			m=self.args.args_d['beta'][0,0]
		else:
			m=panel.mean(panel.Y)	
		#e_norm=self.standardize_variable(panel,self.u,reverse_difference)
		self.Y_st,   self.Y_st_long   = self.standardize_variable(panel,panel.Y,reverse_difference)
		self.X_st,   self.X_st_long   = self.standardize_variable(panel,panel.X,reverse_difference)
		self.XIV_st, self.XIV_st_long = self.standardize_variable(panel,panel.XIV,reverse_difference)
		self.Y_pred_st=cf.dot(self.X_st,self.args.args_d['beta'])
		self.Y_pred=cf.dot(panel.X,self.args.args_d['beta'])	
		self.e_norm_long=self.stretch_variable(panel,self.e_norm)
		self.Y_pred_st_long=self.stretch_variable(panel,self.Y_pred_st)
		self.Y_pred_long=np.dot(panel.input.X,self.args.args_d['beta'])
		self.e_long=panel.input.Y-self.Y_pred_long
		
		Rsq, Rsqadj, LL_ratio,LL_ratio_OLS=stat_functions.goodness_of_fit(self, False, panel)
		Rsq2, Rsqadj2, LL_ratio2,LL_ratio_OLS2=stat_functions.goodness_of_fit(self, True, panel)
		a=0
				
	
	def standardize_variable(self,panel,X,norm=False,reverse_difference=False):
		X=panel.arma_dot.dot(self.AMA_1AR,X,self)
		X=(X+self.re_obj_i.RE(X, panel,False)+self.re_obj_t.RE(X, panel,False))
		if (not panel.Ld_inv is None) and reverse_difference:
			X=cf.dot(panel.Ld_inv,X)*panel.a[3]		
		if norm:
			X=X*self.v_inv05
		X_long=self.stretch_variable(panel,X)
		return X,X_long		
	
	def stretch_variable(self,panel,X):
		N,T,k=X.shape
		m=panel.map
		NT=panel.total_obs
		X_long=np.zeros((NT,k))
		X_long[m]=X
		return X_long
		
		

	def copy_args_d(self):
		return copy_array_dict(self.args.args_d)

	
	def h(self,panel,e,z):
		return h(e, z, panel)
	
	def arma_calc(self,panel, u):
		matrices=set_garch_arch(panel,self.args.args_d, u)
		if matrices is None:
			return None		
		self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA, self.e, self.lnv=matrices
		self.AMA_dict={'AMA_1':None,'AMA_1AR':None,'GAR_1':None,'GAR_1MA':None}		
		return matrices	
	

	

def set_garch_arch(panel,args, u):
	return set_garch_arch_c(panel,args, u)
	if c is None:
		#todo: this might not be neccessary at all
		raise RuntimeError('c-library for sparese inversion not compiled, and the scipy version needs to be fixed')
		m=set_garch_arch_scipy(panel,args)
	else:
		m=set_garch_arch_c(panel,args, u)
	return m
		


def set_garch_arch_c(panel,args,u):
	"""Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
	the diagonal band"""
	N, T, _ = u.shape
	rho=np.insert(-args['rho'],0,1)
	psi=args['psi']
	psi=np.insert(args['psi'],0,0) 

	
	AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, lnv=(
		np.append([1],np.zeros(T-1)),
		np.zeros(T),
		np.append([1],np.zeros(T-1)),
		np.zeros(T),
		np.zeros(u.shape),
		np.zeros(u.shape)
	)
	

	#c.arma_arrays(args['lambda'],rho,-args['gamma'],psi,T,AMA_1,AMA_1AR,GAR_1,GAR_1MA,u,e,lnv)	

	lmbda = args['lambda']
	gmma = -args['gamma']
	lengths = np.array(( N , T , 
							len(lmbda), len(rho), len(gmma), len(psi)), 
							dtype=int)

	cfunct.armas(lengths.ctypes.data_as(CIPT), 
						  lmbda.ctypes.data_as(CDPT), rho.ctypes.data_as(CDPT),
						  gmma.ctypes.data_as(CDPT), psi.ctypes.data_as(CDPT),
						  AMA_1.ctypes.data_as(CDPT), AMA_1AR.ctypes.data_as(CDPT),
						  GAR_1.ctypes.data_as(CDPT), GAR_1MA.ctypes.data_as(CDPT),
						  u.ctypes.data_as(CDPT), 
						  e.ctypes.data_as(CDPT), lnv.ctypes.data_as(CDPT)
						  )		
	

	r=[]
	#Creating nympy arrays with name properties. 
	for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
		r.append((locals()[i],i))
	for i in ['e', 'lnv']:
		r.append(locals()[i])
	return r


def set_garch_arch_scipy(panel,args):
	#after implementing ctypes, the scipy version might be dropped entirely
	p,q,d,k,m=panel.pqdkm
	nW,n=panel.nW,panel.max_T

	AAR=-lag_matr(-panel.I,args['rho'])
	AMA_1AR,AMA_1=solve_mult(args['lambda'], AAR, panel.I)
	if AMA_1AR is None:
		return
	GMA=lag_matr(panel.I*0,args['psi'])
	GAR_1MA,GAR_1=solve_mult(-args['gamma'], GMA, panel.I)
	if GAR_1MA is None:
		return
	r=[]
	for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
		r.append((locals()[i],i))
	return r
	
def solve_mult(args,b,I):
	"""Solves X*a=b for a where X is a banded matrix with 1  and args along
	the diagonal band"""
	import scipy
	n=len(b)
	q=len(args)
	X=np.zeros((q+1,n))
	X[0,:]=1
	X2=np.zeros((n,n))
	w=np.zeros(n)
	r=np.arange(n)	
	for i in range(q):
		X[i+1,:n-i-1]=args[i]
	try:
		X_1=scipy.linalg.solve_banded((q,0), X, I)
		if np.any(np.isnan(X_1)):
			return None,None			
		X_1b=cf.dot(X_1, b)
	except:
		return None,None

	return X_1b,X_1


def add_to_matrices(X_1,X_1b,a,ab,r):
	for i in range(0,len(a)):	
		if i>0:
			d=(r[i:],r[:-i])
			X_1[d]=a[i]
		else:
			d=(r,r)
		X_1b[d]=ab[i]	
	return X_1,X_1b

def lag_matr(L,args):
	k=len(args)
	if k==0:
		return L
	L=1*L
	r=np.arange(len(L))
	for i in range(k):
		d=(r[i+1:],r[:-i-1])
		if i==0:
			d=(r,r)
		L[d]=args[i]

	return L



def copy_array_dict(d):
	r=dict()
	for i in d:
		r[i]=np.array(d[i])
	return r