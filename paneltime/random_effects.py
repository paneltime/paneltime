
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class re_obj:
	def __init__(self,panel):
		"""Following Greene(2012) p. 413-414"""
		self.panel=panel
		self.sigma_u=0
	
	def RE(self,x,recalc=True):
		panel=self.panel
		if panel.FE_RE==0:
			return x
		if panel.FE_RE==1:
			self.xFE=self.FRE(x)
			return self.xFE
		if recalc:
			N,T,k=x.shape
			T_i=self.panel.T_i.reshape(N,1,1)
			self.avg_Tinv=np.mean(1/panel.T_i)
			self.xFE=self.FRE(x)
			self.e_var=self.panel.mean(self.xFE**2)/(1-self.avg_Tinv)
			self.v_var=self.panel.mean(x**2)-self.e_var
			
			self.theta=(1-np.sqrt(self.e_var/(self.e_var+self.v_var*T_i)))*self.panel.included
			
			if np.any(self.theta>1) or np.any(self.theta<0):
				raise RuntimeError("WTF")
		eRE=self.FRE(x,self.theta)
		return eRE
	
	def dRE(self,dx,x,vname):
		"""Returns the first and second derivative of RE"""
		panel=self.panel
		if not hasattr(self,'dxFE'):
			self.dxFE=dict()
			self.dFE_var=dict()
			self.dtheta=dict()
			self.de_var=dict()
			self.dv_var=dict()
	
		if self.panel.FE_RE==0:
			return dx
		elif self.panel.FE_RE==1:
			return self.FRE(dx)	
		(N,T,k)=dx.shape
		T_i=panel.T_i.reshape(N,1,1)		

		self.dxFE[vname]=self.FRE(dx)
		self.de_var[vname]=2*np.sum(np.sum(self.xFE*self.dxFE[vname],0),0)/(self.panel.NT*(1-self.avg_Tinv))
		self.dv_var[vname]=(2*np.sum(np.sum(x*dx,0),0)/self.panel.NT)-self.de_var[vname]
		

		

		self.dtheta_de_var=(-0.5*(1/self.e_var)*(1-self.theta)*self.theta*(2-self.theta))
		self.dtheta_dv_var=(0.5*(T_i/self.e_var)*(1-self.theta)**3)
		self.dtheta[vname]=(self.dtheta_de_var*self.de_var[vname]+self.dtheta_dv_var*self.dv_var[vname])
		
		dRE0=self.FRE(dx,self.theta)
		dRE1=self.FRE(x,self.dtheta[vname],True)
		return (dRE0+dRE1)*self.panel.included
	
	def ddRE(self,ddx,dx1,dx2,x,vname1,vname2):
		"""Returns the first and second derivative of RE"""
		panel=self.panel
		if self.panel.FE_RE==0 or self.sigma_u<0:
			return ddx
		elif self.panel.FE_RE==1:
			return self.FRE(ddx)	
		(N,T,k)=dx1.shape
		(N,T,m)=dx2.shape
		if ddx is None:
			ddxFE=0
			ddx=0
			hasdd=False
		else:
			ddxFE=self.FRE(ddx)
			hasdd=True
			
		dxFE1=self.dxFE[vname1].reshape(N,T,k,1)
		dxFE2=self.dxFE[vname2].reshape(N,T,1,m)
		dx1=dx1.reshape(N,T,k,1)
		dx2=dx2.reshape(N,T,1,m)
		de_var1=self.de_var[vname1].reshape(k,1)
		de_var2=self.de_var[vname2].reshape(1,m)
		dtheta_de_var=self.dtheta_de_var.reshape(N,T,1,1)
		dtheta_dv_var=self.dtheta_dv_var.reshape(N,T,1,1)
		theta=self.theta.reshape(N,T,1,1)
		T_i=panel.T_i.reshape(N,1,1,1)
		

	
		
		d2e_var=2*np.sum(np.sum(dxFE1*dxFE2+self.xFE.reshape(N,T,1,1)*ddxFE,0),0)/(self.panel.NT*(1-self.avg_Tinv))
		d2v_var=(2*np.sum(np.sum(dx1*dx2+x.reshape(N,T,1,1)*ddx,0),0)/self.panel.NT)-d2e_var	
		
		d2theta_d_e_v_var=-0.5*dtheta_dv_var*(1/self.e_var)*(3*(theta-2)*theta-2)
		d2theta_d_v_var =-0.75*(T_i/self.e_var)**2*(1-theta)**5
		d2theta_d_e_var =-0.5*dtheta_de_var*(1/self.e_var)*(4-3*(2-theta)*theta)	
		
		ddtheta =(d2theta_d_e_v_var* de_var1+d2theta_d_v_var   * de_var1) * de_var2
		ddtheta+=(d2theta_d_e_var  * de_var1+d2theta_d_e_v_var * de_var1) * de_var2
		ddtheta+=ddtheta+ (dtheta_de_var*d2e_var+dtheta_dv_var*d2v_var)
		ddtheta=ddtheta
	
		if hasdd:
			dRE00=self.FRE(ddx,self.theta.reshape(N,T,1,1))
		else:
			dRE00=0
		dRE01=self.FRE(dx1,self.dtheta[vname2].reshape(N,T,1,m),True)
		dRE10=self.FRE(dx2,self.dtheta[vname1].reshape(N,T,k,1),True)
		dRE11=self.FRE(x.reshape(N,T,1,1),ddtheta,True)
		return (dRE00+dRE01+dRE10+dRE11)*panel.included.reshape(N,T,1,1)
	
	def FRE(self,x,w=1,d=False):
		"""returns x after fixed effects, and set lost observations to zero"""
		#assumes x is a "N x T x k" matrix
		if x is None:
			return None
		if len(x.shape)==3:
			N,T,k=x.shape
			s=((N,1,k),(N,T,1))
			T_i=self.panel.T_i
		elif len(x.shape)==4:
			N,T,k,m=x.shape
			s=((N,1,k,m),(N,T,1,1))
			T_i=self.panel.T_i.reshape((N,1,1,1))
		ec=x*self.panel.included.reshape(s[1])
		sum_ec=np.sum(ec,1).reshape(s[0])
		sum_ec_all=np.sum(sum_ec,0)	
		dFE=(w*(sum_ec/T_i-sum_ec_all/self.panel.NT))*self.panel.included.reshape(s[1])
		if d==False:
			return ec*self.panel.included.reshape(s[1])-dFE
		else:
			return -dFE