#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains classes used in the regression
from .. import likelihood as logl

from . import arguments
from . import conversion
from .. import functions as fu

import numpy as np
import time




#todo: split up panel in sub classes

class Panel:
	def __init__(self,datainput,settings):

		self.input=datainput
		self.options=settings
		self.dataframe=datainput.dataframe

	def init(self):
		self.initial_defs()
		self.arrayize()
		self.diff_transform()
		self.final_defs()

	def initial_defs(self):
		if np.all(np.var(self.input.Y,0)==0):
			raise RuntimeError("No variation in Y")
		p,q,d,k,m=self.options.pqdkm
		self.orig_size=len(self.input.X)
		self.max_lags=self.input.max_lags
		self.lost_obs = max((p,q,self.max_lags))+max((m,k,self.max_lags))+d#+3
		self.nW,self.nZ,self.n_beta=len(self.input.W.columns),len(self.input.Z.columns),len(self.input.X.columns)
		self.define_h_func()
		self.undiff=None
		if self.input.idvar is None:
			self.options.fixed_random_group_eff=0
		if self.input.idvar is None:
			self.options.fixed_random_group_eff=0		
		self.m_zero = False
		if  m==0 and k>0:
			self.m_zero = True
			k=0
			print("Warning: GARCH term removed since ARCH terms was set to 0")
		self.pqdkm=p,q,d,k,m



	def final_defs(self):
		self.W_a=self.W*self.included[3]
		self.tot_lost_obs=self.lost_obs*self.N
		self.NT=np.sum(self.included[3])
		self.NT_before_loss=self.NT
		if not hasattr(self,'n_dates'):
			self.number_of_RE_coef=0
			self.number_of_RE_coef_in_variance=0
			self.options.fixed_random_group_eff = 0
			self.options.fixed_random_time_eff = 0
			self.options.fixed_random_variance_eff = 0
		else:
			self.number_of_RE_coef=self.N*(self.options.fixed_random_group_eff>0)+self.n_dates*(self.options.fixed_random_time_eff>0)
			self.number_of_RE_coef_in_variance=(self.N*(self.options.fixed_random_group_eff>0)
																														+self.n_dates*(self.options.fixed_random_time_eff>0))*(self.options.fixed_random_variance_eff>0)
		self.args=arguments.arguments(self)
		self.df=self.NT-self.args.n_args-self.number_of_RE_coef-self.number_of_RE_coef_in_variance
		self.set_instrumentals()
		self.tobit()

	def diff_transform(self):
		T=self.max_T
		d=self.pqdkm[2]
		self.I=np.diag(np.ones(T))
		#differencing operator:
		if d==0:
			return
		diff_0=np.diag(np.ones(T-1),-1)
		diff_d=(self.I-diff_0)
		undiff=np.tril(np.ones((T,T)))
		for i in range(1,d):
			diff_d=fu.dot(self.I-diff_0,diff_d)
			undiff=np.cumsum(undiff,0)
		self.undiff=undiff
		#apply the :
		self.X=self.apply_diff(self.X, diff_d, d, True)
		self.Y=self.apply_diff(self.Y, diff_d, d, False)
		if not self.Z is None:
			self.Z=self.apply_diff(self.Z, diff_d, d, True)

	def apply_diff(self,X,diff_d,d,recreate_intercept):
		X_out=fu.dot(diff_d,X)*self.included[3]
		if self.input.has_intercept and recreate_intercept:
			X_out[:,:,0]=1
		X_out[:,:d]=0
		return X_out

	def params_ok(self,args):
		a=self.q_sel,self.p_sel,self.M_sel,self.K_sel
		for i in a:
			if len(i)>0:
				if np.any(np.abs(args[i])>0.999):
					return False
		return True

	def set_instrumentals(self):
		if self.input.Z.shape[1]==1: #The instrument is just a constant
			self.XIV = self.X
		else:
			self.XIV = self.Z
		return


	def subtract_means(self,X,Y,Z):
		subtract=self.options.subtract_means
		if not subtract:
			return X,Y,Z
		X=X-np.mean(X,0)
		if self.input.has_intercept:
			X[:,0]=1
		Y=Y-np.mean(Y,0)
		if self.input.Z.shape[1]==1:
			return X,Y,Z
		Z=Z-np.mean(Z,0)
		Z[:,0]=1		

	def is_single(self):
		idvar,t=self.input.idvar,self.input.timevar
		try:
			if (np.all(idvar.iloc[0]==idvar) or np.all(t.iloc[0]==t)):
				return True
		except:
			return True
		return False


	def arrayize(self):
		"""Splits X and Y into an arry of equally sized matrixes rows equal to the largest for each idvar"""

		X, Y, W, idvar,timevar,Z=[to_numpy(i) for i in 
				(self.input.X, self.input.Y, self.input.W, self.input.idvar,self.input.timevar,self.input.Z)]
		X,Y,Z=self.subtract_means(X,Y,Z)
		
		
		if self.is_single():
			self.arrayize_single(timevar, X, Y, W, Z)			
		else:
			self.arrayize_multi(idvar, timevar, X, Y, W, Z)

		self.cross_section_count()

	def arrayize_single(self, timevar, X, Y, W, Z):
		NT,k=X.shape
		self.total_obs=NT

		if not np.all(timevar[:,0]==np.sort(timevar[:,0])):#remove in production
			raise RuntimeError("The time variable is not sorted!?!")			
		self.X=np.array(X.reshape((1,NT,k)))
		self.Y=np.array(Y.reshape((1,NT,1)))
		NT,kW=W.shape
		self.W=np.array(W.reshape((1,NT,kW)))
		NT,kZ=Z.shape
		self.Z=None
		if Z.shape[1]>1:#instrumental variables used
			self.Z=np.array(Z.reshape((1,NT,kZ)))
		self.time_map=None
		self.map=np.arange(NT).reshape(1,NT)
		self.N=1
		self.max_T=NT
		self.T_arr=np.array([[NT]])
		self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
		self.date_count_mtrx=None
		self.date_count=None
		self.idincl=np.array([True])
		if not np.all(timevar[:,0]==np.sort(timevar[:,0])):#remove in production
			raise RuntimeError("The arrayize procedure has unsorted the time variable!?!")		
		self.masking()
		self.date_map = [(0,t) for t in range(NT)]

	def arrayize_multi(self, idvar, timevar, X, Y, W, Z):
		NT,k=X.shape
		self.total_obs=NT

		sel,ix=np.unique(idvar,return_index=True)
		N=len(sel)
		sel=(idvar.T==sel.reshape((N,1)))
		T=np.sum(sel,1)
		self.max_T=np.max(T)
		self.idincl=T>=self.lost_obs+self.options.min_group_df
		self.X=arrayize(X, N,self.max_T,T, self.idincl,sel)
		self.Y=arrayize(Y, N,self.max_T,T, self.idincl,sel)
		self.W=arrayize(W, N,self.max_T,T, self.idincl,sel)
		self.Z=arrayize(Z, N,self.max_T,T, self.idincl,sel)		
		self.N=np.sum(self.idincl)
		varmap=np.arange(NT).reshape(NT,1)
		self.map=arrayize(varmap, N,self.max_T,T, self.idincl,sel,dtype=int).reshape((self.N,self.max_T))
		self.T_arr=T[self.idincl].reshape((self.N,1))
		self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
		self.masking()
		if len(self.included[3])<5:
			raise RuntimeError(f"{len(self.included[3])} valid observations, cannot perform panel analysis. Try without panel (don't specify ID and time)")
		self.map_time_to_panel_structure(timevar, self.N,T, self.idincl,sel,self.included[3])

		if np.sum(self.idincl)<len(self.idincl):
			idname=self.input.idvar.columns[0]
			idremoved=np.array(self.input.idvar_orig)[ix,0][self.idincl==False]
			s = formatarray(idremoved,90,', ')
			print(f"The following {idname}s were removed because of insufficient observations:\n %s" %(s))
		#remove in production. Checking sorting:
		tvar=arrayize(timevar, N,self.max_T,T, self.idincl,sel)
		a=[tvar[i][0][0] for i in self.date_map]
		if not np.all(a==np.sort(a)):	
			raise RuntimeError("It seems the data is not properly sorted on time")
			
		


	def cross_section_count(self):
		self.T_i=np.sum(self.included[3],1).reshape((self.N,1,1))#number of observations for each i
		self.T_i=self.T_i+(self.T_i<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	
		self.N_t=np.sum(self.included[3],0).reshape((1,self.max_T,1))#number of observations for each t
		self.N_t=self.N_t+(self.N_t<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	


	def masking(self):
		("Creates a mask self.included that is zero for all observations exeeding the last."
   		"This is neccessary in unbalanced panels, since all time series start at row 0 in "
		"the panel data matrices. Series with fewer observations will then end before the last row "
		"and this must be taken into account.\n"
		"self.included is an array of masks")

		included=np.array([(self.date_counter<self.T_arr[i]) for i in range(self.N)])
		zeros=np.zeros((self.N,self.max_T,1))
		ones=np.ones((self.N,self.max_T,1))			
		self.included=[None,None]
		self.zeros=[None,None]
		self.ones=[None,None]

		#self.included[2] is 2 dim, self.included[3] is 3 dim and so on ...:
		self.included.extend([included.reshape(list(included.shape)[:-1]+[1]*i) for i in range(5)])		
		self.zeros.extend([zeros.reshape(list(zeros.shape)[:-1]+[1]*i) for i in range(5)])	
		self.ones.extend([ones.reshape(list(ones.shape)[:-1]+[1]*i) for i in range(5)])	


	def tobit(self):
		"""Sets the tobit threshold"""
		tobit_limits=self.options.tobit_limits
		if tobit_limits is None:
			return
		if len(tobit_limits)!=2:
			print("Warning: The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
		if (not (tobit_limits[0] is None)) and (not( tobit_limits[1] is None)):
			if tobit_limits[0]>tobit_limits[1]:
				raise RuntimeError("floor>ceiling. The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
		g=[1,-1]
		self.tobit_I=[None,None]
		self.tobit_active=[False, False]#lower/upper threshold
		desc=['tobit_low','tobit_high']
		for i in [0,1]:
			self.tobit_active[i]=not (tobit_limits[i] is None)
			if self.tobit_active[i]:
				if np.sum((g[i]*self.Y<g[i]*tobit_limits[i])*self.included[3]):
					print("Warning: there are observations of self.Y outside the non-censored interval. These will be ignored.")
				I=(g[i]*self.Y<=g[i]*tobit_limits[i])*self.included[3]
				self.Y[I]=tobit_limits[i]
				if np.var(self.Y)==0:
					raise RuntimeError("Your tobit limits are too restrictive. All observationss would have been cencored. Cannot run regression with these limits.")
				self.tobit_I[i]=I
				if np.sum(I)>0 and np.sum(I)<self.NT and False:#avoiding singularity #Not sure why this is here, shuld be deleted?
					self.X=np.concatenate((self.X,I),2)
					self.input.X_names.append(desc[i])

	def map_time_to_panel_structure(self,timevar, N,T_count, idincl,sel,incl):
		if timevar is None:
			return None
		N,T,k=incl.shape
		unq,ix=np.unique(timevar,return_inverse=True)
		
		
		t_arr =arrayize(np.array(ix).reshape((len(timevar),1)), 
													 N, self.max_T, T_count, idincl,sel,int)#maps N,T -> unique date
		
		#arrayize creates a NxTx1 array, as self.Y, containing unique indicies for the time variable in a panel structure
		
		grp_cnt=incl*np.arange(N).reshape(N,1,1)
		t_cnt=incl*np.arange(T).reshape(1,T,1)
		incl=incl[:,:,0]
		t=np.concatenate((t_arr[incl],  grp_cnt[incl], t_cnt[incl]),1)
		a=np.argsort(t[:,0])
		t=t[a]#three columns: unique date index, group number, day sequence


		t_id=t[:,0]#unique date
		t_map=[[] for i in range(np.max(t_id)+1)]#all possible unique dates
		for i in range(len(t_id)):
			t_map[t_id[i]].append(t[i,1:])#appends group and day sequence
		
		
		t_map_tuple, tcnt = self.build_time_map(t_map, t_arr)

		#A full random effects calculation is infeasible because of complexity and computing costs. 
		#A quazi random effects weighting is used. It  is more conservative than the full
		#RE weight theta=1-sd_pooled/(sd_pooled+sd_within/T)**0.5
		#If the weights are too generous, the RE adjustment may add in stead of reducing noise. 
		n=len(tcnt)
		self.n_dates=n
		self.date_count=np.array(tcnt).reshape(n,1,1)
		self.date_map=t_map_tuple

		self.dmap_all =(
		[t_map_tuple[i][0] for i in range(self.n_dates)],
		[t_map_tuple[i][1] for i in range(self.n_dates)]
		)   
		
		a=0

	def build_time_map(self, t_map, t_arr):
		"""
		Constructs a structured mapping of unique time indices to corresponding 
		group and time sequence data.

		The output `t_map_tuple` provides an efficient way to retrieve all observations 
		for a specific unique date in a panel data structure. Instead of storing lists 
		of indices separately, this function organizes them as tuples, making it easier 
		to access all relevant observations in `self.Y`, `self.X`, or `t_arr` using 
		structured indexing.

		Example usage:
			- `self.X[t_map_tuple[20]]` retrieves all observations in `self.X` 
			from the 20th unique date in the dataset.

		Args:
			t_map (list of lists): A list where each index represents a unique date,
								and each element contains a list of [group_id, time_sequence]
								for that date.
			t_arr (numpy matrix/array): unique date id_s in a panel data structure

		Returns:
			tuple:
				- t_map_tuple (list of tuples): Each tuple contains:
					* A tuple of all group IDs for a given date
					* A tuple of corresponding time sequences
				- tcnt (list): The count of groups for each unique date.
		"""
		t_map_tuple=[]
		tcnt=[]
		self.date_count_mtrx=np.zeros(t_arr.shape)
		for i in range(len(t_map)):
			a=np.array(t_map[i]).T#group and day sequence for unique date i
			if len(a):
				m=(tuple(a[0]),tuple(a[1]))#group and day sequence reference tuple
				n_t=len(a[0])#number of groups at this unique date
				t_map_tuple.append(m)	#group and day reference tuple for the data matrix, for each unique date
				tcnt.append(n_t) #count of groups at each date
				self.date_count_mtrx[m]=n_t#timeseries matrix of the group count

		return t_map_tuple, tcnt


	def define_h_func(self):
		h_def="""
def h(e,z):
	e2			=	e**2+1e-5
	h_val		=	np.log(e2)	
	h_e_val		=	2*e/e2
	h_2e_val	=	2/e2-4*e**2/e2**2

	return h_val,h_e_val,h_2e_val,None,None,None
		"""	
		h_definition=self.options.h_function
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


	def mean(self,X,axis=None):
		dims=list(X.shape)
		dims[2:]=[1]*(len(dims)-2)
		#X=X*self.included.reshape(dims)
		if axis is None:
			return np.sum(X)/self.NT
		if axis==1:
			dims.pop(1)
			return np.sum(X,1)/self.T_i.reshape(dims)
		if axis==0:
			dims.pop(0)
			return np.sum(X,0)/self.N_t.reshape(dims)
		if axis==(0,1):
			return np.sum(np.sum(X,0),0)/self.NT

	def var(self,X,axis=None,k=1,mean=None,included=None):
		dims=list(X.shape)
		dims_m=np.array(X.shape)
		dims[2:]=[1]*(len(dims)-2)	
		if included is None:
			a=self.included[len(dims)]
		else:
			a=included
		if mean is None:
			m=self.mean(X*a, axis)
		else:
			m=mean

		if axis==None:
			Xm=(X-m)*a
			return np.sum(Xm**2)/(self.NT-k)

		if axis==1:
			dims_m[1]=1
			m=m.reshape(dims_m)
			Xm=(X-m)*a
			dims.pop(1)
			return np.sum((Xm)**2,1)/np.maximum(self.T_i-k,1).reshape(dims)
		if axis==0:
			dims_m[0]=1		
			m=m.reshape(dims_m)
			Xm=(X-m)*a
			dims.pop(0)
			return np.sum((Xm)**2,0)/np.maximum(self.N_t-k,1).reshape(dims)
		if axis==(0,1):
			dims_m[0:2]=1
			m=m.reshape(dims_m)
			Xm=(X-m)*a			
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






def to_numpy(x):
	if x is None:
		return None
	x=np.array(x)
	if len(x.shape)==2:
		return x
	return x.reshape((len(x),1))

def formatarray(array,linelen,sep):
	s=sep.join([str(i) for i in array])
	s='\n'.join(s[n:n + linelen] for n in range(0, len(s), linelen))	
	return s
