
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class re_obj:
	def __init__(self,panel):
		"""Following Greene(2012) p. 413-414"""
		self.panel=panel
		self.sigma_u=0
	
	def RE(self,e,recalc=True):
		if self.panel.FE_RE==0 or (self.sigma_u<0 and (not recalc)):
			return e
		if self.panel.FE_RE==1:
			self.eFE=self.FE(e)
			return self.eFE
		if recalc:
			self.eFE=self.FE(e)
			self.FE_var=self.panel.mean(self.eFE**2)
			N,T,k=e.shape
			self.T_i=self.panel.T_i.reshape(N)
			self.e_grp_mean=self.panel.mean(e,1).reshape(N)
			self.grp_var=np.sum(self.e_grp_mean**2)/N
			self.avg_Tinv=np.mean(1/self.T_i)
			self.FE_tadj=1-self.T_i*self.avg_Tinv
			self.grp_tadj=self.T_i*(1-self.avg_Tinv)
			self.tot_var=(self.FE_tadj*self.FE_var + self.grp_tadj*self.grp_var)
			self.sigma_u=self.grp_var-self.FE_var*self.avg_Tinv*(1-self.avg_Tinv)
			if self.sigma_u<0:
				return e
			self.theta=(1-np.sqrt(self.FE_var/self.tot_var))*self.panel.included
			self.theta=np.maximum(self.theta,1e-15)
			if np.any(self.theta>1):
				raise RuntimeError("WTF")
		eRE=self.FE(e,self.theta)
		return eRE
	
	def dRE(self,de,e,vname):
		"""Returns the first and second derivative of RE"""
		if not hasattr(self,'deFE'):
			self.deFE=dict()
			self.dFE_var=dict()
			self.dtheta=dict()
			self.dgrp=dict()
			self.dgrp_var=dict()
			self.dtot_var=dict()
	
		if self.panel.FE_RE==0 or self.sigma_u<0:
			return de
		elif self.panel.FE_RE==1:
			return self.FE(de)	
		sqrt_expr=np.sqrt(1/(self.grp_var*self.T_i*self.FE_var))
		self.deFE[vname]=self.FE(de)
		self.dFE_var[vname]=2*np.sum(np.sum(self.eFE*self.deFE[vname],0),0)/self.panel.NT
		self.dgrp[vname]=self.panel.mean(de,1)
		self.dgrp_var[vname]=2*np.sum(self.e_grp_mean*self.dgrp[vname],0)/self.panel.N
		self.dtot_var[vname]=(self.FE_tadj*self.self.dFE_var[vname] + self.grp_tadj*self.dgrp_var[vname])
		f=(1/self.FE_var)*self.dFE_var[vname]-(1/self.grp_var)*self.dgrp_var[vname]
		self.dtheta[vname]=0.5*(self.theta-1)*f*self.panel.included
		
		dRE0=self.FE(de,self.theta)
		dRE1=self.FE(e,self.dtheta[vname],True)
		return (dRE0+dRE1)*self.panel.included
	
	def ddRE(self,dde,de1,de2,e,vname1,vname2):
		"""Returns the first and second derivative of RE"""
		if self.panel.FE_RE==0 or self.sigma_u<0:
			return dde
		elif self.panel.FE_RE==1:
			return self.FE(dde)	
		(N,T,k)=de1.shape
		(N,T,m)=de2.shape
		if dde is None:
			ddeFE=0
			ddgrp=0
			hasdd=False
		else:
			ddeFE=self.FE(dde)
			ddgrp=self.panel.mean(dde,1)
			hasdd=True
		eFE=self.eFE.reshape(N,T,1,1)
		ddFE_var=2*np.sum(np.sum(eFE*ddeFE+self.deFE[vname1].reshape(N,T,k,1)*self.deFE[vname2].reshape(N,T,1,m),0),0)/self.panel.NT
		e_grp_mean=self.e_grp_mean.reshape(N,1,1)
		ddgrp_var=2*np.sum(e_grp_mean*ddgrp+self.dgrp[vname1].reshape(N,k,1)*self.dgrp[vname2].reshape(N,1,m),0)/self.panel.N
		ddtot_var=(self.FE_tadj*ddFE_var + self.grp_tadj*ddgrp_var)
		
		ddthetatheta=self.dtheta[vname].reshape(N,k,1)*self.dtheta[vname].reshape(N,1,m)/(1-self.theta)
		ddtheta_FE=((1/self.FE_var)*ddFE_var+(1/(self.FE_var**2))*self.dFE_var[vname1].reshape(k,1)*self.dFE_var[vname2].reshape(1,m))
		ddtheta_tot=((1/self.tot_var)*ddtot_var+(1/(self.tot_var**2))*self.dtot_var[vname1].reshape(k,1)*self.dtot_var[vname2].reshape(1,m))
		ddtheta=ddthetatheta+0.5*(1-self.theta)*(ddtheta_FE+ddtheta_tot)
		ddtheta=ddtheta.reshape(N,1,k,m)*self.panel.included.reshape(N,T,1,1)
	
		if hasdd:
			dRE00=self.FE(dde,self.theta.reshape(N,T,1,1))
		else:
			dRE00=0
		dRE01=self.FE(de1.reshape(N,T,k,1),self.dtheta[vname2].reshape(N,T,1,m),True)
		dRE10=self.FE(de2.reshape(N,T,1,m),self.dtheta[vname1].reshape(N,T,k,1),True)
		dRE11=self.FE(e.reshape(N,T,1,1),ddtheta,True)
		return (dRE00+dRE01+dRE10+dRE11)
	
	def FE(self,e,w=1,d=False):
		"""returns x after fixed effects, and set lost observations to zero"""
		#assumes e is a "N x T x k" matrix
		if e is None:
			return None
		if len(e.shape)==3:
			N,T,k=e.shape
			s=((N,1,k),(N,T,1))
			T_i=self.panel.T_i
		elif len(e.shape)==4:
			N,T,k,m=e.shape
			s=((N,1,k,m),(N,T,1,1))
			T_i=self.panel.T_i.reshape((N,1,1,1))
		ec=e*self.panel.included.reshape(s[1])
		sum_ec=np.sum(ec,1).reshape(s[0])
		sum_ec_all=np.sum(sum_ec,0)	
		dFE=(w*(sum_ec/T_i-sum_ec_all/self.panel.NT))*self.panel.included.reshape(s[1])
		if d==False:
			return ec*self.panel.included.reshape(s[1])-dFE
		else:
			return -dFE