#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object
import sys
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.win-amd64-3.5'))
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.linux-x86_64-3.5'))
try:#only using c function if installed
	import cfunctions as c
except ImportError as e:
	c=None
import numpy as np
import functions as fu
import calculus_functions as cf
import stat_functions as stat
import random_effects as re
from scipy import sparse as sp
import scipy

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
	that store dynamic variables that depend on the \n
	If args is a dictionary, the ARMA-GARCH orders are 
	determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
	with the  panel object
	"""
	def __init__(self,args,panel,constraints=None):
		self.errmsg=''
		self.errmsg_h=''
		self.panel=panel
		self.re_obj_i=re.re_obj(panel,True)
		self.re_obj_t=re.re_obj(panel,False)
		self.LL_const=-0.5*np.log(2*np.pi)*panel.NT
	
		self.args_v=panel.args.conv_to_vector(args)
		if not constraints is None:
			constraints.within(self.args_v,True)	
			constraints.set_fixed(self.args_v)
		self.args_d=panel.args.conv_to_dict(self.args_v)
		self.h_err=""
		self.h_def=panel.h_def
		self.LL=self.LL_calc(panel)
		try:
			self.LL=self.LL_calc(panel)
			
		except Exception as e:
			self.LL=None
			self.errmsg=self.errmsg_h
			if self.errmsg=='':
				self.errmsg=str(e)
		if not self.LL is None:
			if np.isnan(self.LL):
				self.LL=None
		
		


	def LL_calc(self,panel):
		args=self.args_d#using dictionary arguments
		X=panel.X
		matrices=set_garch_arch(panel,args)
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
		e_RE = e+self.re_obj_i.RE(e)+self.re_obj_t.RE(e)
		e_REsq = e_RE**2
		LL = self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
		
		if abs(LL)>1e+100: 
			return None
		self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
		self.u,self.e, self.lnv_ARMA        = u,         e,       lnv_ARMA
		self.lnv,self.v,self.v_inv          = lnv,       v,       v_inv
		self.e_RE,self.e_REsq               = e_RE,      e_REsq

		return LL
	
	def garch(self,panel,args,GAR_1MA,e):
		if panel.m>0:
			h_res=self.h(e, args['z'][0])
			if h_res==None:
				return None
			(self.h_val,     self.h_e_val,
			 self.h_2e_val,  self.h_z_val,
			 self.h_2z_val,  self.h_ez_val)=[i*panel.included for i in h_res]
			return cf.dot(GAR_1MA,self.h_val)
		else:
			(self.h_val,    self.h_e_val,
			 self.h_2e_val, self.h_z_val,
			 self.h_2z_val, self.h_ez_val,
			 self.avg_h)=(0,0,0,0,0,0,0)
			return 0			
	
	def group_variance(self,panel,lnv,e,args):
		N=panel.N
		if panel.m==0 or N==1:
			self.avg_lne2,self.davg_lne2,self.d2avg_lne2   = None,   None,  None
			self.avg_e2, self.zmu =None,None
			return 0
		if False: #using average of h function as group variance
			self.avg_h=panel.mean(self.h_val,1).reshape((N,1,1))*panel.a
			self.avg_lne2,self.davg_lne2,self.d2avg_lne2   = self.avg_h,  self.h_e_val, self.h_2e_val
			self.avg_e2, self.zmu                          = None,  1

		else: #using log average of e**2 function as group variance
			avg_e2=panel.mean(e**2,1).reshape((N,1,1))
			avg_lne2=panel.group_var_wght*np.log(avg_e2)*panel.a
			#the derivative, for later use
			davg_lne2=panel.group_var_wght*(2*e/avg_e2)	

			self.avg_lne2,self.davg_lne2   = avg_lne2,  davg_lne2 
			self.avg_e2, self.zmu          = avg_e2, 0	
			
		return args['mu'][0]*self.avg_lne2


	def standardize(self):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self"""
		sd_inv=self.v_inv**0.5
		panel=self.panel
		m=panel.lost_obs
		N,T,k=panel.X.shape
		Y=cf.dot(self.AMA_1AR,panel.Y)
		Y=(Y+self.re_obj_i.RE(Y,False)+self.re_obj_t.RE(Y,False))*sd_inv
		X=cf.dot(self.AMA_1AR,panel.X)
		X=(X+self.re_obj_i.RE(X,False)+self.re_obj_t.RE(X,False))*sd_inv
		self.e_st=self.e_RE*sd_inv
		self.Y_st=Y
		self.X_st=X
		incl=panel.included.reshape(N,T)
		self.e_st_long=self.e_st[incl,:]
		self.Y_st_long=self.Y_st[incl,:]
		self.X_st_long=self.X_st[incl,:]

	def copy_args_d(self):
		return fu.copy_array_dict(self.args_d)

	
	def h(self,e,z):
		d={'e':e,'z':z}
		try:
			exec(self.h_def,globals(),d)
		except Exception as err:
			if self.h_err!=str(err):
				self.errmsg_h="Warning,error in the ARCH error function h(e,z): %s" %(err)
			h_err=str(e)
			return None
	
		return d['ret']	

def set_garch_arch(panel,args):
	if c is None:
		m=set_garch_arch_scipy(panel,args)
	else:
		m=set_garch_arch_c(panel,args)
	return m
		
		
def set_garch_arch_c(panel,args):
	"""Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
	the diagonal band"""
	n=panel.max_T
	rho=np.insert(-args['rho'],0,1)
	psi=np.insert(args['psi'],0,0)

	r=np.arange(n)
	AMA_1,AMA_1AR,GAR_1,GAR_1MA=(
	    np.diag(np.ones(n)),
		np.zeros((n,n)),
		np.diag(np.ones(n)),
		np.zeros((n,n))
	)
	c.bandinverse(args['lambda'],rho,-args['gamma'],psi,n,AMA_1,AMA_1AR,GAR_1,GAR_1MA)
	return  AMA_1,AMA_1AR,GAR_1,GAR_1MA


def set_garch_arch_scipy(panel,args):

	p,q,m,k,nW,n=panel.p,panel.q,panel.m,panel.k,panel.nW,panel.max_T

	AAR=-lag_matr(-panel.I,args['rho'])
	AMA_1AR,AMA_1=solve_mult(args['lambda'], AAR, panel.I)
	if AMA_1AR is None:
		return
	GMA=lag_matr(panel.zero,args['psi'])	
	GAR_1MA,GAR_1=solve_mult(-args['gamma'], GMA, panel.I)
	if GAR_1MA is None:
		return
	return AMA_1,AMA_1AR,GAR_1,GAR_1MA
	
def solve_mult(args,b,I):
	"""Solves X*a=b for a where X is a banded matrix with 1  and args along
	the diagonal band"""
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
		L[d]=args[i]

	return L
