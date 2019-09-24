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
import system_arguments as arguments

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
	that store dynamic variables that depend on the \n
	
	"""
	def __init__(self,args,panel,constraints=None):
		self.errmsg=''
		self.LL_const=-0.5*np.log(2*np.pi)*panel.NT
		
		self.LL_calc(args,panel,constraints)
		try:
			self.LL_calc(args,panel,constraints)
			
		except Exception as e:
			self.errhandling(e)

		if not self.LL is None:
			if np.isnan(self.LL):
				self.LL=None
				
				
	def LL_calc(self,args_d,panel,constraints):	
		args_v=panel.args.system_conv_to_vector(args_d)
		if not constraints is None:
			constraints.within(args_v,True)	
			constraints.set_fixed(args_v)
		self.args_d=panel.args.system_conv_to_dicts(args_v)	
		self.args_v=args_v
		lls, v_inv,lnv,e_RE=[],[],[],[]
		for i in range(panel.args.n_equations):
			ll=calc(self.args_d, panel, constraints,i)
			if ll is None:
				return None
			lls.append(ll)
			v_inv.append(ll.v_inv)
			lnv.append(ll.lnv)
			e_RE.append(ll.e_RE)

		s=system(panel,self.args_d['rho'],e_RE,v_inv,lnv)
		self.LL = self.LL_const-0.5*np.sum((lnv+s.rho_lndet+s.e_rho_e)*panel.included)
		
		self.system = s
		self.lls = lls
				
				
	def errhandling(self,e):
		self.LL=None
		for i in self.lls:
			if i.errmsg_h!='':
				self.errmsg=i.errmsg_h
				break
		if self.errmsg=='':
			self.errmsg=str(e)
	
		
	


class system():
	def __init__(self,panel,rho,e_RE,v_inv,lnv):
		self.rho=rho
		self.panel=panel
		self.rho_properties()
		e_RE,v_inv,lnv=np.array(e_RE),np.array(v_inv),np.array(lnv)
		self.v_inv=v_inv
		self.e_RE=e_RE
		self.n_eq=e_RE.shape[0]
		self.data_shape=e_RE.shape
		self.e_norm=e_RE*v_inv
		self.e_rho_prod()	
		#gradient:
		self.dLL_e=-self.e_rho*v_inv*self.panel.included
		self.dLL_lnv=-0.5*(1-self.e_rho*self.e_norm)*self.panel.included
		self.hessian()
		
	def hessian(self):
		shape=[self.n_eq]+list(self.data_shape)
		d2LL_de2=np.zeros(shape)
		self.d2LL_dln_de=np.zeros(shape)
		self.d2LL_de_dln=np.zeros(shape)
		self.d2LL_dln2=np.zeros(shape)
		
		for i in range(self.n_eq):
			for j in range(self.n_eq):
				d2LL_de2[i,j]=-self.rho_inv[i,j]*self.v_inv[i]*self.v_inv[j]*self.panel.included
				self.d2LL_dln_de[i,j]=0.5*self.e_RE[i]*self.rho_inv[i,j]*self.v_inv[i]*self.v_inv[j]*self.panel.included
				self.d2LL_de_dln[i,j]=0.5*self.e_RE[j]*self.rho_inv[i,j]*self.v_inv[i]*self.v_inv[j]*self.panel.included
				self.d2LL_dln2[i,j]= - 0.25*self.e_RE[j]*self.e_RE[i]*self.rho_inv[i,j]*self.v_inv[i]*self.v_inv[j]*self.panel.included
			self.d2LL_dln_de[i,i]+=0.5*self.e_rho[i]*self.v_inv[i]*self.panel.included
			self.d2LL_de_dln[i,i]+=0.5*self.e_rho[i]*self.v_inv[i]*self.panel.included
			self.d2LL_dln2[i,i]+=0.5*self.e_rho[i]*self.e_norm[i]*self.panel.included
		
		
	def e_rho_prod(self):
		e_rho=np.zeros(self.data_shape)
		e_rho_e=np.zeros(self.data_shape[1:])
		for i in range(self.n_eq):
			for j in range(self.n_eq):
				e_rho[i]+=self.e_norm[j]*self.rho_inv[j,i]
		for i in range(self.n_eq):
			e_rho_e+=e_rho[i]*self.e_norm[i]
		self.e_rho=np.array(e_rho)
		self.e_rho_e=np.array(e_rho_e)
		
		
	def rho_properties(self):
		self.rho_inv=np.linalg.inv(self.rho)
		self.rho_det=np.linalg.det(self.rho)
		self.rho_lndet=np.log(self.rho_det)
		self.rho_cofactors=np.zeros(self.rho.shape)
		if len(self.rho)>1:
			for i in range(len(self.rho)):
				for j in range(i):
					c=np.delete(self.rho,i,0)
					c=np.delete(c,j,1)
					self.rho_cofactors[i,j]=np.linalg.det(c)
					self.rho_cofactors[j,i]=self.rho_cofactors[i,j]		
					
	def rho_derivatives(self):
		for i in range(self.n_eq):
			for i in range(i):
				a=0
				
				
				
def cofactor(X,m,k,mmap=None):
	if not mmap is None:
		nz=np.nonzero(mmap==str([m,k]))
		n=len(nz[0])
		m=nz[0][0]
		k=nz[1][0]
	c=np.delete(X,m,0)
	c=np.delete(c,k,1)
	cf=n*(-1)**(m+k)*np.linalg.det(c)
		
	
	
	
	
		

class calc:
	"""Calculates variables neccesary for calculating the log likelihood"""
	def __init__(self,args,panel,constraints,i):

		self.re_obj=re.re_obj(panel)
		self.args_d=args
		self.h_err=""
		self.calc(panel,i)
		self.id=i


	def calc(self,panel,i):
		args=self.args_d[i]#using dictionary arguments
		X=panel.X[i]
		Y=panel.Y[i]
		matrices=set_garch_arch(panel,args)
		if matrices is None:
			return None		
		
		AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
		(N,T,k)=X.shape

		u = Y-cf.dot(X,args['beta'])
		e = cf.dot(AMA_1AR,u)
		lnv_ARMA = self.garch(panel, args, GAR_1MA,e)
		W_omega = cf.dot(panel.W_a,args['omega'])
		lnv = W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
		grp = self.group_variance(panel, lnv, e,args)
		lnv+=grp
		lnv = np.maximum(np.minimum(lnv,100),-100)
		v = np.exp(lnv)*panel.a
		v_inv = np.exp(-lnv)*panel.a	
		e_RE = self.re_obj.RE(e)
		e_REsq = e_RE**2
		self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
		self.u,self.e, self.lnv_ARMA        = u,         e,       lnv_ARMA
		self.lnv,self.v,self.v_inv          = lnv,       v,       v_inv
		self.e_RE,self.e_REsq               = e_RE,      e_REsq

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
		e2=e**2
		zeroe2=(e2==0)
		lne2=np.log(e2+ze2)
		self.lne2=(lne2-panel.mean(lne2))*panel.included
		#the derivative, for later use
		d_lne2=2*(zeroe2/(e+zeroe2))
		self.d_lne2=(d_lne2-panel.mean(d_lne2))*panel.included
		dd_lne2=-2*(zeroe2/(e**2+zeroe2))
		self.d_lne2=(dd_lne2-panel.mean(dd_lne2))*panel.included

		return self.re_obj_i_vol.RE(self.lne2)


	def standardize(self):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self"""
		sd_inv=self.v_inv**0.5
		panel=self.panel
		m=panel.lost_obs
		N,T,k=panel.X.shape
		Y=cf.dot(self.AMA_1AR,panel.Y)
		Y=self.re_obj.RE(Y,False)*sd_inv
		X=cf.dot(self.AMA_1AR,panel.X)
		X=self.re_obj.RE(X,False)*sd_inv
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
		try:
			d=dict()
			exec(self.panel.h_def,globals(),d)
			return d['h'](e,z)
		except Exception as err:
			if self.h_err!=str(err):
				self.errmsg_h="Warning,error in the ARCH error function h(e,z): %s" %(err)
			h_err=str(e)

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
