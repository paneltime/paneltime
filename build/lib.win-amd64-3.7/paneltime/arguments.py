#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains the argument class for the panel object

import stat_functions as stat
import numpy as np
import functions as fu
import loglikelihood as logl




class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,panel):
		p, q, d, k, m=panel.pqdkm
		self.categories=['beta','rho','lambda','gamma','psi','omega']
		if panel.z_active:
			self.categories+=['z']
		self.mu_removed=True
		if not self.mu_removed:
			self.categories+=['mu']
		
		self.panel=panel
		self.make_namevector(panel,p, q, k, m)
		self.set_init_args()
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
		#beta=stat.OLS_simple(panel.input.Y,panel.input.X,True,False)
		self.init_e_st=e[panel.included]
		self.init_e_st=self.init_e_st/np.var(self.init_e_st)**0.5
		args['beta']=beta
		if panel.settings.group_fixed_random_eff.value==0 or panel.settings.variance_fixed_random_eff==0:
			args['omega'][0]=np.log(panel.var(e))

	
		self.args_start=fu.copy_array_dict(args)
		args_old=self.panel.input.args_archive.args
		if (not args_old is None) and (not default): 
			for name,ar_type in [['beta',False],
					  ['omega',False],
					  ['rho',True],
					  ['lambda',False],
					  ['psi',False],
					  ['gamma',True]]:
				self.insert_arg(name,args,ar_type)
			if panel.z_active:
				self.insert_arg('z',args)
			if not self.mu_removed:
				self.insert_arg('mu',args)
			
		self.args_init=args
		self.set_restricted_args(p, d, q, m, k,panel,e,beta)
		

	def set_restricted_args(self,p, d, q, m, k, panel,e,beta):
		self.args_restricted=self.initargs(p, d, q, m, k, panel)
		self.args_OLS=self.initargs(p, d, q, m, k, panel)		
		self.args_restricted['beta'][0][0]=np.mean(panel.Y)
		self.args_restricted['omega'][0][0]=np.log(np.var(panel.Y))
		self.args_OLS['beta']=beta
		self.args_OLS['omega'][0][0]=np.log((np.var(e*panel.included)*len(e[0])/np.sum(panel.included)))
	
	def create_null_ll(self):
		if not hasattr(self,'LL_OLS'):
			self.LL_OLS=logl.LL(self.args_OLS,self.panel).LL
			self.LL_null=logl.LL(self.args_restricted,self.panel).LL	
		
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
		
	def insert_arg(self,argname,args,AR_type=False):
		arg=args[argname]
		arg_old=self.panel.input.args_archive.args[argname]
		names=self.names_d[argname]
		names_old=self.panel.input.args_archive.names_d[argname]
		n=min((len(arg),len(arg_old)))
		for i in names_old:
			if not i in names:
				return
		if n==0:
			return
		if len(arg.shape)==2:
			arg[:,0]=0
		else:
			arg[:]=0
		if not AR_type or len(arg_old)<=n:
			for i in range(len(names)):
				if names[i] in names_old:
					arg[i]=arg_old[names_old.index(names[i])]
		else:
			arg[:n-1]=arg_old[:n-1]
			arg[n-1]=np.sum(arg_old[n-1:])
	
			
def add_names(T,namsestr,category,d,names):
	a=[]
	for i in range(T):
		a.append(namsestr %(i,))
	names.extend(a)
	d[category]=a