
#!/usr/bin/env python
# -*- coding: utf-8 -*-

class REObj:
	def __init__(self,panel):
		"""Following Greene(2012) p. 413-414"""
		self.panel=panel
	
	def RE(self,e,recalc=True):
		
		if self.panel.FE_RE==0:
			return e
		if self.panel.FE_RE==1:
			return self.eFE
		if recalc:
			self.eFE=self.FE(e)
			self.vLSDV=np.sum(self.eFE**2)/self.panel.NT_afterloss
			self.theta=(1-np.sqrt(self.vLSDV/(self.panel.grp_v*self.panel.n_i)))*self.panel.included
			self.theta=np.maximum(self.theta,0)
			if np.any(self.theta>1):
				raise RuntimeError("WTF")
		eRE=self.FE(e,self.theta)
		return eRE
	
	def dRE(self,de,e,vname):
		"""Returns the first and second derivative of RE"""
		if not hasattr(self,'deFE'):
			self.deFE=dict()
			self.dvLSDV=dict()
			self.dtheta=dict()
	
		if self.panel.FE_RE==0:
			return de
		elif self.panel.FE_RE==1:
			return self.FE(self.panel,de)	
		sqrt_expr=np.sqrt(1/(self.panel.grp_v*self.panel.n_i*self.vLSDV))
		self.deFE[vname]=self.FE(self.panel,de)
		self.dvLSDV[vname]=np.sum(np.sum(self.eFE*self.deFE[vname],0),0)/self.panel.NT_afterloss
		self.dtheta[vname]=-sqrt_expr*self.dvLSDV[vname]*self.panel.included
	
		dRE0=self.FE(self.panel,de,self.theta)
		dRE1=self.FE(self.panel,e,self.dtheta[vname],True)
		return (dRE0+dRE1)*self.panel.included
	
	def ddRE(self,dde,de1,de2,e,vname1,vname2):
		"""Returns the first and second derivative of RE"""
		if self.panel.FE_RE==0:
			return dde
		elif self.panel.FE_RE==1:
			return self.FE(self.panel,dde)	
		(N,T,k)=de1.shape
		(N,T,m)=de2.shape
		if dde is None:
			ddeFE=0
			hasdd=False
		else:
			ddeFE=self.FE(self.panel,dde)
			hasdd=True
		eFE=self.eFE.reshape(N,T,1,1)
		ddvLSDV=np.sum(np.sum(eFE*ddeFE+self.deFE[vname1].reshape(N,T,k,1)*self.deFE[vname2].reshape(N,T,1,m),0),0)/self.panel.NT_afterloss
	
		ddtheta1=(np.sqrt(1/(self.panel.grp_v*self.panel.n_i*(self.vLSDV**2))))*self.dvLSDV[vname1].reshape(k,1)*self.dvLSDV[vname2].reshape(1,m)
		ddtheta2=ddtheta1+(-np.sqrt(1/(self.panel.grp_v*self.panel.n_i*self.vLSDV)))*ddvLSDV
		ddtheta=ddtheta2.reshape(N,1,k,m)*self.panel.included.reshape(N,T,1,1)
	
		if hasdd:
			dRE00=self.FE(self.panel,dde,self.theta.reshape(N,T,1,1))
		else:
			dRE00=0
		dRE01=self.FE(self.panel,de1.reshape(N,T,k,1),self.dtheta[vname2].reshape(N,T,1,m),True)
		dRE10=self.FE(self.panel,de2.reshape(N,T,1,m),self.dtheta[vname1].reshape(N,T,k,1),True)
		dRE11=self.FE(self.panel,e.reshape(N,T,1,1),ddtheta,True)
		return (dRE00+dRE01+dRE10+dRE11)
	
	def FE(self,e,w=1,d=False):
		"""returns x after fixed effects, and set lost observations to zero"""
		#assumes e is a "N x T x k" matrix
		if e is None:
			return None
		if len(e.shape)==3:
			N,T,k=e.shape
			s=((N,1,k),(N,T,1))
			n_i=self.panel.n_i
		elif len(e.shape)==4:
			N,T,k,m=e.shape
			s=((N,1,k,m),(N,T,1,1))
			n_i=self.panel.n_i.reshape((N,1,1,1))
		ec=e*self.panel.included.reshape(s[1])
		sum_ec=np.sum(ec,1).reshape(s[0])
		sum_ec_all=np.sum(sum_ec,0)	
		dFE=-(w*(sum_ec/n_i-sum_ec_all/self.panel.NT_afterloss))*self.panel.included.reshape(s[1])
		if d==False:
			return ec*self.panel.included.reshape(s[1])+dFE
		else:
			return dFE