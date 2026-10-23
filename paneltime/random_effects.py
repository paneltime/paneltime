#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class REObj:
	def __init__(self,panel,group,T_i,T_i_count,fixed_random_eff):
		"""Following Greene(2012) p. 413-414"""
		if fixed_random_eff==0:
			self.fe_re=0
			return
		self.sigma_u=0
		self.group=group
		self.avg_Tinv=1/np.mean(T_i_count) #T_i_count is the total number of observations for each group (N,1)
		other_dimension_count = panel.date_count if group else panel.T_i
		self.avg_Ninv=1/np.mean(other_dimension_count)
		self.T_i=T_i*panel.included[3]#self.T_i is T_i_count at each observation (N,T,1)
		self.fe_re=fixed_random_eff

	
	def RE(self,x,panel,recalc=True):
		if self.fe_re==0:
			return np.zeros(x.shape)
		if self.fe_re==1:
			self.xFE=self.FRE(x,panel)
			return self.xFE
		if self.avg_Tinv==1:
			return None
		if recalc:
			N,T,k=x.shape

			incl=panel.included[3]


			mu=panel.mean(incl*x) #grand (weighted) mean of x; ~0 for residuals u, but not in general
			self.xFE=(x+self.FRE(x,panel))*incl
			#subtract mu explicitly so e_var/v_var are correct variances (not raw second moments)
			#for non-zero-mean x. Reduces to the original formulas when mu~=0.
			mu_FE = panel.mean(incl*self.xFE)
			#CHANGED: previously e_var was the one-way within variance /(1-avg_Tinv) and
			#v_var = total variance - e_var. That let the other dimension's effect variance leak into e_var
			#(time effects into the group pass and vice versa), distorting theta and the variance components.
			#Now: Two-way (Swamy-Arora style): the other dimension's effect is removed before estimating
			#both the idiosyncratic and own-dimension variance, so neither leaks into the other.
			m_own=self.FRE(x,panel,means_only=True,group=self.group)
			m_oth=self.FRE(x,panel,means_only=True,group=not self.group)
			x_w=(x-m_own-m_oth+mu)*incl
			#degrees of freedom as in plm's Swamy-Arora: the K slope coefficients are not free, N+T-1 effects are removed
			K=panel.X.shape[2]-int(panel.input.has_intercept)
			dof=1-self.avg_Tinv-self.avg_Ninv+(1-K)/panel.NT
			self.e_var=panel.mean(incl*x_w**2)/dof
			x_adj=(x-m_oth+mu)*incl
			m_adj=self.FRE(x_adj,panel,means_only=True)
			n_own=panel.NT*self.avg_Tinv#number of own-dimension units
			n_df=max(n_own-1-K,1)#between regression: own units less intercept and K slopes
			self.v_var=panel.mean(incl*(m_adj-mu)**2)*n_own/n_df-self.e_var*self.avg_Tinv
			self._x, self._x_w, self._m_adj, self._mu, self._dof, self._n_own, self._n_df = x, x_w, m_adj, mu, dof, n_own, n_df#used by dRE
			if self.v_var<0:
				#print("Warning, negative group random effect variance. 0 is assumed")
				self.v_var=0
				self.theta=panel.zeros[3]
				return np.zeros(x.shape)
			denom=self.e_var+self.v_var*self.T_i
			if (not np.isfinite(self.e_var) or self.e_var<=0
					or not np.all(np.isfinite(denom)) or np.any(denom<=0)):
				raise FloatingPointError("Random-effects variance components produce an undefined theta")
			self.theta=(1-np.sqrt(self.e_var/denom))*panel.included[3]
			self.theta*=(self.T_i>1)
			if np.any(self.theta>1) or np.any(self.theta<0):
				raise RuntimeError("WTF")
		#x=panel.Y
		eRE=self.FRE(x,panel,self.theta)
		return eRE

	def _dvars(self,dx,panel):
		"""First derivatives of the two-way e_var and v_var, and the demeaned dx terms they are built from."""
		incl=panel.included[3]
		dmu=np.sum(dx,axis=(0,1))/panel.NT
		dm_own=self.FRE(dx,panel,means_only=True,group=self.group)
		dm_oth=self.FRE(dx,panel,means_only=True,group=not self.group)
		dx_w=(dx-dm_own-dm_oth+dmu)*incl
		de_var=2*np.sum(self._x_w*dx_w,axis=(0,1))/(panel.NT*self._dof)
		dq=(self.FRE((dx-dm_oth+dmu)*incl,panel,means_only=True)-dmu)*incl
		n_own=self._n_own
		c=n_own/(self._n_df*panel.NT)
		dv_var=2*c*np.sum((self._m_adj-self._mu)*dq,axis=(0,1))-de_var*self.avg_Tinv
		return dx_w,dq,de_var,dv_var,c

	def dRE(self,dx,panel):
		"""Derivative of RE(x) for dx=dx/dparam (N,T,k), where x is the input of the last RE(x) call with recalc=True.
		Includes the effect of the parameter on theta through e_var and v_var (restored, rewritten for the two-way variances)."""
		if self.fe_re==0:
			return np.zeros(dx.shape)
		if self.fe_re==1:
			return self.FRE(dx,panel)
		if self.v_var==0:
			return np.zeros(dx.shape)
		incl=panel.included[3]
		dx=dx*incl
		_,_,de_var,dv_var,_=self._dvars(dx,panel)
		dtheta_de_var=-0.5*(1/self.e_var)*(1-self.theta)*self.theta*(2-self.theta)
		dtheta_dv_var=0.5*(self.T_i/self.e_var)*(1-self.theta)**3
		dtheta=(dtheta_de_var*de_var+dtheta_dv_var*dv_var)*(self.T_i>1)*incl
		return (self.FRE(dx,panel,self.theta)+self.FRE(self._x,panel,dtheta))*incl

	def ddRE(self,dx,panel):
		"""Second derivative (N,T,k,k) of RE(x) when x is linear in the parameters (ddx=0), dx=dx/dparam (N,T,k).
		Restored for the two-way variances; only the theta terms contribute since FRE is linear in x."""
		N,T,k=dx.shape
		incl4=panel.included[4]
		if self.fe_re!=2 or self.v_var==0:
			return np.zeros((N,T,k,k))
		incl=panel.included[3]
		dx=dx*incl
		dx_w,dq,de,dv,c=self._dvars(dx,panel)
		th=self.theta
		e=self.e_var
		s=1-th
		th_e=-0.5*(1/e)*s*th*(2-th)
		th_v=0.5*(self.T_i/e)*s**3
		th_ev=-0.5*th_v*(1/e)*(3*(th-2)*th+2)
		th_vv=-0.75*(self.T_i/e)**2*s**5
		th_ee=-0.5*th_e*(1/e)*(4-3*(2-th)*th)
		d2e=2*np.einsum('ntk,ntl->kl',dx_w,dx_w)/(panel.NT*self._dof)
		d2v=2*c*np.einsum('ntk,ntl->kl',dq,dq)-d2e*self.avg_Tinv
		mask=(self.T_i>1)*incl
		dtheta=(th_e*de+th_v*dv)*mask
		th4=lambda a:a.reshape(N,T,1,1)
		d2theta=(th4(th_ee)*np.outer(de,de)+th4(th_ev)*(np.outer(de,dv)+np.outer(dv,de))
				+th4(th_vv)*np.outer(dv,dv)+th4(th_e)*d2e+th4(th_v)*d2v)*th4(mask)
		#out[:,:,i,j]: FRE(dx_i,dtheta_j)+FRE(dx_j,dtheta_i)+FRE(x,d2theta_ij)
		t1=np.stack([self.FRE(dx,panel,dtheta[:,:,j:j+1]) for j in range(k)],axis=3)
		t3=np.stack([self.FRE(self._x,panel,d2theta[:,:,:,j]) for j in range(k)],axis=3)
		return (t1+np.swapaxes(t1,2,3)+t3)*incl4

	#deleted calc_theta: helper for ddRE (replaced by the theta derivatives inside ddRE)

	def ddRE_general(self,dx1,dx2,ddx,panel):
		"""Second derivative (N,T,k,m) of RE(x) with respect to two parameter blocks, where
		dx1=dx/dp1 (N,T,k), dx2=dx/dp2 (N,T,m) and ddx=d2x/dp1dp2 (N,T,k,m) or None if zero.
		x is the input of the last RE(x) call with recalc=True."""
		N,T,k=dx1.shape
		m=dx2.shape[2]
		incl4=panel.included[4]
		if self.fe_re==0 or (self.fe_re==2 and self.v_var==0):
			return np.zeros((N,T,k,m))
		if self.fe_re==1:
			if ddx is None:
				return np.zeros((N,T,k,m))
			return np.stack([self.FRE(ddx[:,:,:,j],panel) for j in range(m)],axis=3)*incl4
		incl=panel.included[3]
		dx1=dx1*incl
		dx2=dx2*incl
		w1,q1,de1,dv1,c=self._dvars(dx1,panel)
		w2,q2,de2,dv2,_=self._dvars(dx2,panel)
		d2e=2*np.einsum('ntk,ntl->kl',w1,w2)/(panel.NT*self._dof)
		d2v=2*c*np.einsum('ntk,ntl->kl',q1,q2)-d2e*self.avg_Tinv
		if ddx is not None:
			ddx=ddx*incl4
			_,_,de12,dv12,_=self._dvars(ddx.reshape(N,T,k*m),panel)
			d2e=d2e+de12.reshape(k,m)
			d2v=d2v+dv12.reshape(k,m)
		th=self.theta
		e=self.e_var
		s=1-th
		th_e=-0.5*(1/e)*s*th*(2-th)
		th_v=0.5*(self.T_i/e)*s**3
		th_ev=-0.5*th_v*(1/e)*(3*(th-2)*th+2)
		th_vv=-0.75*(self.T_i/e)**2*s**5
		th_ee=-0.5*th_e*(1/e)*(4-3*(2-th)*th)
		mask=(self.T_i>1)*incl
		dtheta1=(th_e*de1+th_v*dv1)*mask
		dtheta2=(th_e*de2+th_v*dv2)*mask
		th4=lambda a:a.reshape(N,T,1,1)
		d2theta=(th4(th_ee)*np.outer(de1,de2)+th4(th_ev)*(np.outer(de1,dv2)+np.outer(dv1,de2))
				+th4(th_vv)*np.outer(dv1,dv2)+th4(th_e)*d2e+th4(th_v)*d2v)*th4(mask)
		#out[:,:,i,j]: FRE(dx1_i,dtheta2_j)+FRE(dx2_j,dtheta1_i)+FRE(x,d2theta_ij)+FRE(ddx_ij,theta)
		out=np.stack([self.FRE(dx1,panel,dtheta2[:,:,j:j+1]) for j in range(m)],axis=3)
		out+=np.stack([self.FRE(dx2,panel,dtheta1[:,:,i:i+1]) for i in range(k)],axis=2)
		out+=np.stack([self.FRE(self._x,panel,d2theta[:,:,:,j]) for j in range(m)],axis=3)
		if ddx is not None:
			out+=np.stack([self.FRE(ddx[:,:,:,j],panel,self.theta) for j in range(m)],axis=3)
		return out*incl4

	def FRE(self,x,panel,w=1,d=False, means_only=False, group = None):
		if group is None:
			group = self.group
		if group:
			return self.FRE_group(x,w,d,panel, means_only)
		else:
			return self.FRE_time(x,w,d,panel, means_only)

	def FRE_group(self,x,w,d,panel, means_only=False):
		"""returns x after fixed effects, and set lost observations to zero"""
		#assumes x is a "N x T x k" matrix
		if x is None:
			return None
		T_i,s=get_subshapes(panel,x,True)
		incl=panel.included[len(s)]

		sum_x=np.sum(x*incl,1).reshape(s)
		mean_x=sum_x/T_i
		if means_only:
			return np.broadcast_to(mean_x, x.shape).copy()
		mean_x_all=np.sum(sum_x,0)/panel.NT
		try:
			dFE=w*(mean_x_all-mean_x)*incl#last product expands the T vector to a TxN matrix
		except (RuntimeWarning,OverflowError) as e:
			print(e)
			remove_extremes([w])
			dFE=w*(mean_x_all-mean_x)*incl#last product expands the T vector to a TxN matrix
			remove_extremes([dFE])
		return dFE

	def FRE_time(self,x,w,d,panel, means_only=False):
		#assumes x is a "N x T x k" matrix


		if x is None:
			return None
		mean_x,mean_x_all,incl=mean_time(panel, x)
		if means_only:
			return np.broadcast_to(mean_x, x.shape).copy()
		try:
			dFE=(w*(mean_x_all-mean_x))*incl#last product expands the T vector to a TxN matrix
		except (RuntimeWarning,OverflowError) as e:
			print(e)
			remove_extremes([w])
			dFE=(w*(mean_x_all-mean_x))*incl#last product expands the T vector to a TxN matrix		
			remove_extremes([dFE])
		return dFE


def mean_time(panel,x,mean_dates=False):
	#todo: fix fast so that it allways worked
	#Currently, the fast excepts on unbalanced panels bc ut maps a ragged matrix
	#possibel solution, change def of panel.dmap_all so that fills with zeros the non entries
	#Problem with this is that there currently is no element on x that is allways zero.
	try:
		return mean_time_fast(panel,x,mean_dates)
	except ValueError as e:
		return mean_time_slow(panel,x,mean_dates)



def mean_time_slow(panel,x,mean_dates=False):
	n_dates=panel.n_dates
	dmap=panel.date_map
	date_count,s=get_subshapes(panel,x,False)
	incl=panel.included[len(s)]
	x=x*incl
	sum_x_dates=np.zeros(s)
	for i in range(n_dates):
		sum_x_dates[i]=np.sum(x[dmap[i]],0)		
	mean_x_dates=sum_x_dates/date_count
	if mean_dates:
		return mean_x_dates
	mean_x=np.zeros(x.shape)
	for i in range(n_dates):
		mean_x[dmap[i]]=mean_x_dates[i]	
	mean_x_all=np.sum(sum_x_dates,0)/panel.NT
	return mean_x,mean_x_all,incl

def mean_time_fast(panel,x,mean_dates=False):
	#at present works only on balanced panels
	dmap_all=panel.dmap_all
	date_count,s=get_subshapes(panel,x,False)
	incl=panel.included[len(s)]
	x=x*incl
	sum_x_dates=np.zeros(s)

	N,T,k = x.shape

	sum_x_dates = np.sum(x[dmap_all],1).reshape((panel.n_dates,1,k))

	mean_x_dates=sum_x_dates/date_count
	if mean_dates:
		return mean_x_dates
	mean_x=np.zeros(x.shape)
	mean_x[dmap_all]=mean_x_dates	
	mean_x_all=np.sum(sum_x_dates,0)/panel.NT
	return mean_x,mean_x_all,incl



def get_subshapes(panel,x,group):
	if group:
		if len(x.shape)==3:
			N,T,k=x.shape
			s=(N,1,k)
			T_i=panel.T_i
		elif len(x.shape)==4:
			N,T,k,m=x.shape
			s=(N,1,k,m)
			T_i=panel.T_i.reshape((N,1,1,1))	
		return T_i,s
	else:
		date_count=panel.date_count
		n_dates=panel.n_dates
		if len(x.shape)==3:
			N,T,k=x.shape
			s=(n_dates,1,k)

		elif len(x.shape)==4:
			N,T,k,m=x.shape
			s=(n_dates,1,k,m)
			date_count=date_count.reshape((n_dates,1,1,1))	
		return date_count,s			



def remove_extremes(args,max_arg=1e+100):
	for i in range(len(args)):
		s=np.sign(args[i][np.abs(args[i])>max_arg])
		args[i][np.abs(args[i])>max_arg]=s*max_arg