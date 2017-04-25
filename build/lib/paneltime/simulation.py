#!/usr/bin/env python
# -*- coding: utf-8 -*-

#simulalte a panel data GARCH model

import numpy as np
import regprocs as rp
import regobj
import functions as fu


class simulation:
	"""Creates an object that can simulate ARIMA-GARCH-timeseries data"""
	def __init__(self,N,T,beta,rho=[0.5],lmbda=[0.5],psi=[0.5],gamma=[0.5],omega=1,mu=1,z=1,residual_sd=1,group_sd=1):

		self.args,self.p,self.q,self.m,self.k,self.beta_len=self.new_args(beta,rho,lmbda,psi,gamma,omega,mu,z)
		self.T=T
		self.N=N
		self.residual_sd=residual_sd
		self.group_sd=group_sd
		self.max_lags=np.max((self.p,self.q,self.m,self.k))
		self.L=rp.make_lag_matrices(T,self.max_lags)
		self.I=np.diag(np.ones(T))
		self.zero=self.I*0		
		self.AAR_1MA, self.GAR_1MA=matrices=self.set_garch_arch()
		
	def sim(self):
		args=self.args
		eRE=np.random.normal(0,self.residual_sd,(self.N,self.T,1))+np.random.normal(0,self.group_sd,(self.N,1,1))
		
		u=fu.dot(self.AAR_1MA,eRE)
		
	
		if self.m>0:
			h=eRE**2+args['z']
			group_eff=np.random.normal(0,1,(self.N,1,1))*args['mu']
			lnv=fu.dot(self.GAR_1MA,h)+group_eff+args['omega']
		else:
			lnv_ARMA=0	
			
		v=np.exp(lnv)
		v_inv=np.exp(-lnv)
		
		eRE_GARCH=eRE*v
		
		X=np.random.normal(0,1,(self.N,self.T,self.beta_len-1))
		X=np.concatenate((np.ones((self.N,self.T,1)),X),2)
		X=fu.dot(self.AAR_1MA,X)*v
		Y_pred=fu.dot(X,args['beta'])
		Y=Y_pred+eRE_GARCH
		groups=np.ones((self.N,self.T,1))*np.arange(self.T).reshape((1,self.T,1))
		X=reshape(X,self.max_lags+1)
		Y=reshape(Y,self.max_lags+1)
		groups=reshape(groups,self.max_lags+1)
		return X,Y,groups

	def new_args(self,beta,rho,lmbda,psi,gamma,omega,mu,z):
		p=len(rho)
		q=len(lmbda)
		m=len(psi)
		k=len(gamma)
		beta_len=len(beta)		
		args=dict()
		args['beta']=np.array(beta).reshape((beta_len,1))
		args['omega']=omega
		args['rho']=np.array(rho)
		args['lambda']=np.array(lmbda)
		args['psi']=np.array(psi)
		args['gamma']=np.array(gamma)
		if m>0:
			args['mu']=mu
			args['z']=z
		else:
			args['mu']=0
			args['z']=0

		return args,p,q,m,k,beta_len

	
		
		
	def set_garch_arch(self):
		args,p,q,m,k=self.args,self.p,self.q,self.m,self.k
		AMA=self.I+rp.lag_matr(self.L,self.zero,q,args['lambda'])
		X=self.I-rp.lag_matr(self.L,self.zero,p,args['rho'])
		AAR_1=np.linalg.inv(X)
		AAR_1MA=fu.dot(AAR_1,AMA)
		X=self.I-rp.lag_matr(self.L,self.zero,k,args['gamma'])
		GAR_1=np.linalg.inv(X)
		GMA=rp.lag_matr(self.L,self.zero,m,args['psi'])	
		GAR_1MA=fu.dot(GAR_1,GMA)
		return AAR_1MA, GAR_1MA
	


def reshape(X,n):
	X=X[:,n:,:]
	return np.concatenate(X,0)
	



