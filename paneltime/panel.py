#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains classes used in the regression

import stat_functions as stat
import numpy as np
import time
import threading
import debug
import functions as fu
import calculus_functions as cf
import model_parser
import calculus
import copy
import loglikelihood as logl
from scipy import sparse as sp
import random_effects as re

NON_NUMERIC_TAG='|~|'
min_AC=0.000001


def posdef(a,da):
	return list(range(a,a+da)),a+da

class panel:
	def __init__(self,dataframe,datainput,settings,pqdkm):

		self.input=datainput
		self.settings=settings
		self.dataframe=dataframe
		self.pqdkm=pqdkm
		self.initial_defs()
		self.arrayize()
		self.masking()
		self.lag_variables()
		self.final_defs()
		

	def initial_defs(self):
		if np.all(np.var(self.input.Y,0)==0):
			raise RuntimeError("No variation in Y")
		p,q,d,k,m=self.pqdkm
		self.lost_obs=np.max((p,q))+max((m,k))+d#+3
		self.nW,self.n_beta=self.input.W.shape[1],self.input.X.shape[1]
		self.define_h_func()
		if self.input.IDs_name is None:
			self.settings.group_fixed_random_eff.value=0
		if self.input.timevar is None:
			self.settings.group_fixed_random_eff.value=0		
		self.m_zero = False
		if  m==0 and k>0:
			self.m_zero = True
			p,q,d,k,m=self.pqdkm
			self.pqdkm=p,q,d,k,1
			
	def masking(self):
		
		#"initial observations" mask: 
		self.a=np.array([self.date_counter<self.T_arr[i] for i in range(self.N)])# sets observations that shall be zero to zero by multiplying it with the arrayized variable
	
		#"after lost observations" masks: 
		self.T_i=np.sum(self.included,1).reshape((self.N,1,1))#number of observations for each i
		self.T_i=self.T_i+(self.T_i<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	
		self.N_t=np.sum(self.included,0).reshape((1,self.max_T,1))#number of observations for each i
		self.N_t=self.N_t+(self.N_t<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	
		self.group_var_wght=1-1/np.maximum(self.T_i-1,1)
		
	def final_defs(self):
		self.W_a=self.W*self.a
		self.tot_lost_obs=self.lost_obs*self.N
		self.NT=np.sum(self.included)
		self.NT_before_loss=self.NT+self.tot_lost_obs				
		self.number_of_RE_coef=self.N*(self.settings.group_fixed_random_eff.value>0)+self.n_dates*(self.settings.time_fixed_random_eff.value>0)
		self.number_of_RE_coef_in_variance=(self.N*(self.settings.group_fixed_random_eff.value>0)
											+self.n_dates*(self.settings.time_fixed_random_eff.value>0))*(self.settings.variance_fixed_random_eff.value>0)
		self.args=arguments(self, )
		self.df=self.NT-self.args.n_args-self.number_of_RE_coef-self.number_of_RE_coef_in_variance
		a=0

	def lag_variables(self):
		T=self.max_T
		d=self.pqdkm[2]
		self.I=np.diag(np.ones(T))
		self.zero=np.zeros((T,T))		

		#differencing:
		if d==0:
			return
		L0=np.diag(np.ones(T-1),-1)
		Ld=(self.I-L0)
		for i in range(1,d):
			Ld=cf.dot(self.I-L0,Ld)		
		self.Y=cf.dot(Ld,self.Y)*self.a	
		self.X=cf.dot(Ld,self.X)*self.a
		if self.input.has_intercept:
			self.X[:,:,0]=1
		self.Y[:,:d]=0
		self.X[:,:d]=0	

	def params_ok(self,args):
		a=self.q_sel,self.p_sel,self.M_sel,self.K_sel
		for i in a:
			if len(i)>0:
				if np.any(np.abs(args[i])>0.999):
					return False
		return True


	def arrayize(self):
		"""Splits X and Y into an arry of equally sized matrixes rows equal to the largest for each IDs"""
		X, Y, W, IDs=self.input.X, self.input.Y, self.input.W, self.input.IDs
		timevar=self.input.timevar
		NT,k=X.shape
		if IDs is None:
			self.X=X.reshape((1,NT,k))
			self.Y=Y.reshape((1,NT,1))
			for i in [0,1]:
				if self.input.tobit_active[i]:
					self.tobit_I=self.input.tobit_I[i].reshape((1,NT,1))
			NTW,k=W.shape
			self.W=W.reshape((1,NT,k))
			self.time_map=None
			self.N=1
			self.max_T=NT
			self.T_arr=np.array([[NT]])
			self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
			self.included=np.array([(self.date_counter>=self.lost_obs)*(self.date_counter<self.T_arr[i]) for i in range(self.N)])
		else:
			sel,ix=np.unique(IDs,return_index=True)
			N=len(sel)
			sel=(IDs.T==sel.reshape((N,1)))
			T=np.sum(sel,1)
			self.max_T=np.max(T)
			idincl=T>self.lost_obs+self.settings.min_group_df.value
			self.X=arrayize(X, N,self.max_T,T, idincl,sel)
			self.Y=arrayize(Y, N,self.max_T,T, idincl,sel)
			self.tobit_I=[None,None]
			for i in [0,1]:
				self.tobit_I[i]=arrayize(self.input.tobit_I[i], N,self.max_T,T, idincl,sel,dtype=bool)
			self.W=arrayize(W, N,self.max_T,T, idincl,sel)
			self.N=np.sum(idincl)
			self.T_arr=T[idincl].reshape((self.N,1))
			self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
			self.included=np.array([(self.date_counter>=self.lost_obs)*(self.date_counter<self.T_arr[i]) for i in range(self.N)])
			self.get_time_map(timevar, self.N,T, idincl,sel)
			
	
			
			if np.sum(idincl)<len(idincl):
				idname=self.input.IDs_name[0]
				if idname + NON_NUMERIC_TAG in self.dataframe:
					id_orig=self.dataframe[idname + NON_NUMERIC_TAG]
					idremoved=id_orig[ix,0][idincl==False]
				else:
					idremoved=(self.dataframe[idname])[ix,0][idincl==False]
				s=fu.formatarray(idremoved,90,', ')
				print(f"Warning: The following {idname}s were removed because of insufficient observations:\n %s" %(s))
		self.allzeros=np.zeros((self.N,self.max_T,1))

	
	def get_time_map(self,timevar, N,T_count, idincl,sel):
		if timevar is None:
			return None
		incl=self.included
		N,T,k=incl.shape
		unq,ix=np.unique(timevar,return_inverse=True)
		t=arrayize(np.array(ix).reshape((len(timevar),1)), 
				   N,self.max_T,T_count, idincl,sel,int)#maps N,T -> unique date
		grp_cnt=incl*np.arange(N).reshape(N,1,1)
		t_cnt=incl*np.arange(T).reshape(1,T,1)
		incl=incl[:,:,0]
		t=np.concatenate((t[incl],  grp_cnt[incl], t_cnt[incl]),1)
		a=np.argsort(t[:,0])
		t=t[a]#three columns: unique date index, group number, day sequence
		
	
		tid=t[:,0]#unique date
		t_map=[[] for i in range(np.max(tid)+1)]#all possible unique dates
		for i in range(len(tid)):
			t_map[tid[i]].append(t[i,1:])#appends group and day sequence
		t_map_tuple=[]
		tcnt=[]
		self.date_count_mtrx=np.zeros((N,T,1))
		for i in range(len(t_map)):
			a=np.array(t_map[i]).T#group and day sequence for unique date i
			if len(a):
				m=(tuple(a[0]),tuple(a[1]))#group and day sequence reference tuple
				n_t=len(a[0])#number of groups at this unique date
				t_map_tuple.append(m)	#group and day sequence reference tuple, for each unique date
				tcnt.append(n_t) #count of groups at each date
				self.date_count_mtrx[m]=n_t#timeseries matrix of the group count
				
		
		#A full random effects calculation is infeasible because of complexity and computing costs. 
		#A quazi random effects weighting is used. It  is more conservative than the full
		#RE weight theta=1-sd_pooled/(sd_pooled+sd_within/T)**0.5
		#If the weights are too generous, the RE adjustment may add in stead of reducing noise. 
		n=len(tcnt)
		self.n_dates=n
		self.date_count=np.array(tcnt).reshape(n,1,1)
		self.date_map=t_map_tuple
		
		
	def get_time_map2(self,timevar, N,T_count, idincl,sel):
		if timevar is None:
			return None
		unq,ix=np.unique(timevar,return_inverse=True)
		n_dates=len(unq)
		t=arrayize(np.array(ix).reshape((n_dates,1)), N,self.max_T,T_count, idincl,sel,int)#maps N,T -> unique date
		N,T,k=t.shape

		
		t_map=[[] for i in range(n_dates)]#all possible unique dates
		for i in range(len(tid)):
			t_map[tid[i]].append(t[i,1:])#appends group and day sequence
		t_map_tuple=[]
		tcnt=[]
		self.date_count_mtrx=np.zeros((N,T,1))
		for i in range(len(t_map)):
			a=np.array(t_map[i]).T#group and day sequence for unique date i
			if len(a):
				m=(tuple(a[0]),tuple(a[1]))#group and day sequence reference tuple
				n_t=len(a[0])#number of groups at this unique date
				t_map_tuple.append(m)	#group and day sequence reference tuple, for each unique date
				tcnt.append(n_t) #count of groups at each date
				self.date_count_mtrx[m]=n_t#timeseries matrix of the group count
				
		
		#A full random effects calculation is infeasible because of complexity and computing costs. 
		#A quazi random effects weighting is used. It  is more conservative than the full
		#RE weight theta=1-sd_pooled/(sd_pooled+sd_within/T)**0.5
		#If the weights are too generous, the RE adjustment may add in stead of reducing noise. 
		n=len(tcnt)
		self.n_dates=n
		self.date_count=np.array(tcnt).reshape(n,1,1)
		self.date_map=t_map_tuple
	
	
	def define_h_func(self):
		h_def="""
def h(e,z):
	e2			=	e**2+1e-5
	h_val		=	np.log(e2)	
	h_e_val		=	2*e/e2
	h_2e_val	=	2/e2-4*e**2/e2**2

	return h_val,h_e_val,h_2e_val,None,None,None
		"""	
		h_definition=self.settings.h_function.value
		if h_definition is None:
			h_definition=h_def
		d=dict()
		try:
			exec(h_definition,globals(),d)
			ret=d['h'](1,1)
			if len(ret)!=6:
				raise RuntimeError("""Your custom h-function must return exactly six arguments
				(h, dh/dx and ddh/dxdx for both e and z. the z return values can be set to None)""")
			self.h_def=h_definition
		except Exception as e:
			print('Something is wrong with your custom function, default is used:'+ str(e))
			exec(h_def,globals(),d)
			self.h_def=h_def
		
		self.z_active=True
		for i in ret[3:]:
			self.z_active=self.z_active and not (i is None)	
		if not self.settings.user_constraints.value is None:
			if not self.z_active and 'z' in self.settings.user_constraints.value:
				self.settings.user_constraints.value.pop('z')
			

		
		
	def mean(self,X,axis=None):
		dims=list(X.shape)
		dims[2:]=[1]*(len(dims)-2)
		#X=X*self.included.reshape(dims)
		if axis==None:
			return np.sum(X)/self.NT
		if axis==1:
			dims.pop(1)
			return np.sum(X,1)/self.T_i.reshape(dims)
		if axis==0:
			dims.pop(0)
			return np.sum(X,0)/self.N_t.reshape(dims)
		if axis==(0,1):
			return np.sum(np.sum(X,0),0)/self.NT
			
	def var(self,X,axis=None,k=1,mean=None):
		dims=list(X.shape)
		dims_m=np.array(X.shape)
		dims[2:]=[1]*(len(dims)-2)	
		#X=X*self.included.reshape(dims)
		if mean is None:
			m=self.mean(X, axis)
		else:
			m=mean

		if axis==None:
			return np.sum((X-m)**2)/(self.NT-k)
		count=[]
		if axis==1:
			dims_m[1]=1
			dims.pop(1)
			m=m.reshape(dims_m)
			Xm=(X-m)#*self.included.reshape(dims)			
			return np.sum((Xm)**2,1)/np.maximum(self.T_i-k,1).reshape(dims)
		if axis==0:
			dims_m[0]=1		
			dims.pop(0)
			m=m.reshape(dims_m)
			Xm=(X-m)#*self.included.reshape(dims)			
			return np.sum((Xm)**2,0)/np.maximum(self.N_t-k,1).reshape(dims)
		if axis==(0,1):
			dims_m[0:2]=1
			m=m.reshape(dims_m)
			Xm=(X-m)#*self.included.reshape(dims)			
			return np.sum((Xm)**2,axis)/(self.NT-k)
	
def arrayize(X,N,max_T,T,idincl,sel,dtype=None):
	if X is None:
		return None
	NT,k=X.shape
	if dtype is None:
		Xarr=np.zeros((N,max_T,k))
	else:
		Xarr=np.zeros((N,max_T,k),dtype=dtype)
	T_used=[]
	k=0
	for i in range(len(sel)):
		if idincl[i]:
			Xarr[k,:T[i]]=X[sel[i]]
			k+=1
	Xarr=Xarr[:k]
	return Xarr





class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,panel):
		args=panel.input.args
		p, q, d, k, m=panel.pqdkm
		self.categories=['beta','rho','lambda','gamma','psi','omega']
		if panel.z_active:
			self.categories+=['z']
		self.mu_removed=True
		if not self.mu_removed:
			self.categories+=['mu']
		
		self.args_old=args
		self.panel=panel
		self.set_init_args()
		self.make_namevector(panel,p, q, k, m)
		self.position_defs()
		self.args_v=self.conv_to_vector(self.args_init)
		self.n_args=len(self.args_v)
		

	def initargs(self,p,d,q,m,k,panel):

		args=dict()
		args['beta']=np.zeros((panel.X.shape[2],1))
		args['omega']=np.zeros((panel.W.shape[2],1))
		args['rho']=np.zeros(p)
		args['lambda']=np.zeros(q)
		args['psi']=np.zeros(m)
		args['gamma']=np.zeros(k)
		args['omega'][0][0]=0
		args['mu']=np.array([])
		args['z']=np.array([])			
		if panel.m_zero and k>0:
			args['psi'][0]=1e-8
		
		if m>0 and panel.z_active:
			args['z']=np.array([1e-09])	

		if panel.N>1 and not self.mu_removed:
			args['mu']=np.array([0.0001])			
			

		return args

	def set_init_args(self,default=False):
		panel=self.panel
		p, q, d, k, m=panel.pqdkm
		
		args=self.initargs(p, d, q, m, k, panel)

		#de2=np.roll(e**2,1)-e**2
		#c=stat.correl(np.concatenate((np.roll(de2,1),de2),2),panel)[0,1]

		beta,e=stat.OLS(panel,panel.X,panel.Y,return_e=True)
		self.init_e_st=e[panel.included]
		self.init_e_st=self.init_e_st/np.var(self.init_e_st)**0.5
		args['beta']=beta
		if panel.settings.group_fixed_random_eff.value==0:
			args['omega'][0]=np.log(panel.var(e))

	
		self.args_start=fu.copy_array_dict(args)
		if (not self.args_old is None) and (not default): 
			args['beta']=insert_arg(args['beta'],self.args_old['beta'])
			args['omega']=insert_arg(args['omega'],self.args_old['omega'])
			args['rho']=insert_arg(args['rho'],self.args_old['rho'],True)
			args['lambda']=insert_arg(args['lambda'],self.args_old['lambda'])
			args['psi']=insert_arg(args['psi'],self.args_old['psi'])
			args['gamma']=insert_arg(args['gamma'],self.args_old['gamma'],True)
			if panel.z_active:
				args['z']=insert_arg(args['z'],self.args_old['z'])
			if not self.mu_removed:
				args['mu']=insert_arg(args['mu'],self.args_old['mu'])
			
		self.args_init=args
		self.set_restricted_args(p, d, q, m, k,panel,e,beta)
		

	def set_restricted_args(self,p, d, q, m, k, panel,e,beta):
		self.args_restricted=self.initargs(p, d, q, m, k, panel)
		self.args_OLS=self.initargs(p, d, q, m, k, panel)		
		self.args_restricted['beta'][0][0]=np.mean(panel.Y)
		self.args_restricted['omega'][0][0]=np.log(np.var(panel.Y))
		self.args_OLS['beta']=beta
		self.args_OLS['omega'][0][0]=np.log((np.var(e*panel.included)*len(e[0])/np.sum(panel.included)))
		
	def position_defs(self):
		"""Defines positions in vector argument"""

		self.positions=dict()
		self.positions_map=dict()#a dictionary of indicies containing the string name and sub-position of index within the category
		k=0
		for i in self.categories:
			n=len(self.args_init[i])
			rng=range(k,k+n)
			self.positions[i]=rng
			for j in rng:
				self.positions_map[j]=[0,i,j-k]#equation,category,position
			k+=n
	
	def conv_to_dict(self,args):
		"""Converts a vector argument args to a dictionary argument. If args is a dict, it is returned unchanged"""
		if type(args)==dict:
			return args
		if type(args)==list:
			args=np.array(args)			
		d=dict()
		k=0
		for i in self.categories:
			n=len(self.positions[i])
			rng=range(k,k+n)
			d[i]=args[rng]
			if i=='beta' or i=='omega':
				d[i]=d[i].reshape((n,1))
			k+=n
		return d


	def conv_to_vector(self,args):
		"""Converts a dict argument args to vector argument. if args is a vector, it is returned unchanged.\n
		If args=None, the vector of self.args_init is returned"""
		if type(args)==list or type(args)==np.ndarray:
			return np.array(args)
		v=np.array([])
		for i in self.categories:
			s=args[i]
			if type(s)==np.ndarray:
				s=s.flatten()
			v=np.concatenate((v,s))
		return v


	def make_namevector(self,panel,p, q, k, m):
		"""Creates a vector of the names of all regression varaibles, 
		including variables, ARIMA and GARCH terms. This defines the positions
		of the variables througout the estimation."""
		d=dict()
		names=panel.input.x_names[:]#copy variable names
		d['beta']=list(names)
		add_names(p,'rho%s    AR    p','rho',d,names)
		add_names(q,'lambda%s MA    q','lambda',d,names)
		add_names(k,'gamma%s  GARCH k','gamma',d,names)
		add_names(m,'psi%s    ARCH  m','psi',d,names)
		
		
		d['omega']=panel.input.W_names
		names.extend(panel.input.W_names)
		if m>0:
			if panel.N>1 and not self.mu_removed:
				d['mu']=['mu (var.ID eff.)']
				names.extend(d['mu'])
			if panel.z_active:
				d['z']=['z in h(e,z)']
				names.extend(d['z'])
			
		self.names_v=names
		self.names_d=d


			

			
def add_names(T,namsestr,category,d,names):
	a=[]
	for i in range(T):
		a.append(namsestr %(i,))
	names.extend(a)
	d[category]=a
	

def insert_arg(arg,insert,AR_type=False):
	n=min((len(arg),len(insert)))
	if not AR_type or len(insert)<=n or n==0:
		arg[:n]=insert[:n]
	else:
		arg[:n-1]=insert[:n-1]
		arg[n-1]=np.sum(insert[n-1:])
	return arg