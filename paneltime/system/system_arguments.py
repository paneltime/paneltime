#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains the arguments class used to handle regression arguments

import numpy as np
import functions as fu
import stat_functions as stat


class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,panel, args_d_old):
		p,q,d,k,m=panel.pqdkm
		self.args_d_old=args_d_old
		self.categories=['beta','rho','lambda','gamma','psi','omega','z']
		self.args_init,self.args_d_OLS, self.args_d_restricted=dict(),dict(),dict()
		self.panel=panel
		self.equations=[]
		self.n_equations=len(panel.X)
		self.n_args=[]
		self.positions=dict()
		self.positions_map=dict()
		self.name_positions_map=dict()
		arg_count=0
		self.names_v=[]
		self.eq_number_v=[]
		if args_d_old is None:
			args_d_old=[None]*self.n_equations
		for i in range(self.n_equations):
			e=equation(panel.X[i],panel.Y[i],panel.W,self, args_d_old[i],i,arg_count,panel.X_names[i])
			self.equations.append(e)
			self.n_args.append(e.n_args)
			self.args_init[i]=e.args_init
			self.args_d_OLS[i]=e.args_d_OLS
			self.args_d_restricted[i]=e.args_d_restricted	
			arg_count+=e.n_args
		self.args_init['rho']=np.diag(np.ones(self.n_equations))
		self.args_d_OLS['rho']=np.diag(np.ones(self.n_equations))
		self.args_d_restricted['rho']=np.diag(np.ones(self.n_equations))
		self.n_args_eq=arg_count
		self.n_args_tot=int(np.sum(self.n_args)+(self.n_equations-1)*self.n_equations/2)
		add_rho_names(self.names_v,arg_count)
		self.eq_number_v.extend([None]*(self.n_args_tot-arg_count))
		
		
	def system_conv_to_dicts(self,args):
		args_d=dict()
		if type(args[0])==dict:
			return args
		for eq in self.equations:
			d=dict()
			for category in self.categories:
				rng=eq.positions[category]
				s=self.args_init[eq.id][category].shape
				d[category]=args[rng].reshape(s)
				
			args_d[eq.id]=d
		args_d['rho']=rho_list_to_matrix(args[self.n_args_eq:],self.n_equations)
		return args_d
			
	def system_conv_to_vector(self,args):
		args_v=[]
		if type(args[0])!=dict:
			return args
		n=0
		for i in range(self.n_equations):
			for name in self.categories:
				args_v.extend(args[i][name].flatten())
				n+=len(args[i][name])
		args_v.extend(rho_matrix_to_list(args['rho'],self.n_equations))
		args_v=np.array(args_v)
		return args_v		
		

		
	def rho_definitions(self):
		n=self.n_args_eq
		self.rho_position_list=[]
		r=range(n)
		x=[[[min((i,j)),max((i,j))] for i in r] for j in r]
		self.rho_position_matrix=np.array([[str(x[i,j]) for i in r] for j in r])
		for i in range(n):
			for j in range(i,n):
				self.names_v.append('System reg. rho(%s,%s)' %(i,j))
				self.rho_position_list.append[x[i,j]]
	

	def rho_list_to_matrix(self,lst):
		n=len(self.rho_position_list)
		m=np.zeros((n,n))
		for k in range(n):
			i,j=self.rho_position_list[k]
			m[i,j]=lst[k]
			m[j,i]=lst[k]
		return m
		
	def rho_matrix_to_list(self,m):
		n=len(self.rho_position_list)
		lst=np.zeros(n)
		for k in range(n):
			i,j=self.rho_position_list[k]
			lst[k]=m[i,j]
		return lst
	
	
	
class equation:
	def __init__(self,X,Y,W,arguments,args_d_old,i,arg_count,X_names):
		a=arguments
		self.id=i
		p,q,d,k,m=panel.pqdkm
		self.args_init,self.args_d_OLS, self.args_d_restricted=set_init_args(X,Y,W,args_d_old,p, d, q, m, k,a.panel)
		self.names_d=get_namevector(a.panel,p, q, m, k,X_names,a,i)
		
		self.position_defs(a,arg_count,X_names)
		
		self.args_v=conv_to_vector(self.args_init,a.categories)
		self.n_args=len(self.args_v)
		self.args_rng=range(arg_count,arg_count+self.n_args)
		a.eq_number_v.extend([i]*self.n_args)
		
		
				
	def position_defs(self,system,arg_count,X_names):
		"""Defines positions in vector argument in each equation for the system args_v vector"""
		self.positions_map=dict()#a dictionary of indicies containing the string name and sub-position of index within the category
		self.positions=dict()#a dictionary of category strings containing the index range of the category
		self.beta_map=dict()
		k=arg_count
		for category in system.categories:
			n=len(self.args_init[category])
			rng=range(k,k+n)
			self.positions[category]=rng#self.positions[<category>]=range(<system position start>,<system position end>)
			if category in system.positions:
				system.positions[category].append(rng)
			else:
				system.positions[category]=[rng]
			for j in rng:
				self.positions_map[j]=[category,j-k]#self.positions_map[<system position>]=<category>,<equation position>
				system.positions_map[j]=[self.id,category,j-k]#system.positions_map[<system position>]=<equation number>,<category>,<equation position>
			k+=n
		
		for i in range(len(X_names)):
			self.beta_map[X_names[i]]=self.positions['beta'][i]		
			






	
	
def initargs(X,Y,W,args_old,p,d,q,m,k,panel):
	N,T,k=X.shape
	if args_old is None:
		armacoefs=0
	else:
		armacoefs=0
	args=dict()
	args['beta']=np.zeros((k,1))
	args['omega']=np.zeros((W.shape[2],1))
	args['rho']=np.ones(p)*armacoefs
	args['lambda']=np.ones(q)*armacoefs
	args['psi']=np.ones(m)*armacoefs
	args['gamma']=np.ones(k)*armacoefs
	args['z']=np.array([])	
	if m>0 and N>1:
		args['omega'][0][0]=0
	if m>0:
		args['psi'][0]=0.00001
		args['z']=np.array([0.00001])	

	return args

def set_init_args(X,Y,W,args_old,p,d,q,m,k,panel):
	
	args=initargs(X,Y,W,args_old,p, d, q, m, k, panel)
	args_restricted=fu.copy_array_dict(args)
	if panel.has_intercept:
		args_restricted['beta'][0][0]=panel.mean(Y)
		args_restricted['omega'][0][0]=np.log(panel.var(Y))
	else:
		args_restricted['omega'][0][0]=np.log(panel.var(Y,k=0,mean=0))

	beta,e=stat.OLS(panel,X,Y,return_e=True)
	args['beta']=beta
	args['omega'][0]=np.log(np.sum(e**2*panel.included)/np.sum(panel.included))
	args_OLS=fu.copy_array_dict(args)
	if panel.m_zero:
		args['omega'][0]=0
	
	if not args_old is None: 
		args['beta']=insert_arg(args['beta'],args_old['beta'])
		args['omega']=insert_arg(args['omega'],args_old['omega'])
		args['rho']=insert_arg(args['rho'],args_old['rho'])
		args['lambda']=insert_arg(args['lambda'],args_old['lambda'])
		args['psi']=insert_arg(args['psi'],args_old['psi'])
		args['gamma']=insert_arg(args['gamma'],args_old['gamma'])
		args['z']=insert_arg(args['z'],args_old['z'])
		
	return args,args_OLS, args_restricted
		

	



def conv_to_dict(args,categories,positions):
	"""Converts a vector argument args to a dictionary argument. If args is a dict, it is returned unchanged"""
	if type(args)==dict:
		return args
	else:
		d=dict()
		k=0
		for i in categories:
			n=len(positions[i])
			rng=range(k,k+n)
			d[i]=args[rng]
			if i=='beta' or i=='omega':
				d[i]=d[i].reshape((n,1))
			k+=n
	return d


def conv_to_vector(args,categories):
	"""Converts a dict argument args to vector argument. if args is a vector, it is returned unchanged.\n
	If args=None, the vector of self.args is returned"""
	if type(args)==list or type(args)==np.ndarray:
		return args
	v=np.array([])
	for category in categories:
		s=args[category]
		if type(s)==np.ndarray:
			s=s.flatten()
		v=np.concatenate((v,s))
	return v


def get_namevector(panel,p, q, m, k,X_names,system,eq_num):
	"""Creates a vector of the names of all regression varaibles, 
	including variables, ARIMA and GARCH terms. This defines the positions
	of the variables througout the estimation."""

	names_d=dict()
	#sequence must match definition of categories in arguments.__init__:
	#self.categories=['beta','rho','lambda','gamma','psi','omega','z']
	eq_prefix='%02d|' %(eq_num,)
	names_v=[eq_prefix+i for i in X_names]#copy variable names
	names_d['beta']=names_v
	add_names(p,eq_prefix+'AR term %s (p)','rho',names_d,names_v)
	add_names(q,eq_prefix+'MA term %s (q)','lambda',names_d,names_v)
	add_names(m,eq_prefix+'MACH term %s (m)','psi',names_d,names_v)
	add_names(k,eq_prefix+'ARCH term %s (k)','gamma',names_d,names_v)
	
	names_d['omega']=[eq_prefix+i for i in panel.heteroscedasticity_factors]#copy variable names
	names_v.extend(names_d['omega'])
	if m>0:
		names_d['z']=[eq_prefix+'z in h(e,z)']
		names_v.extend(names_d['z'])
		
	n=len(system.names_v)
	for i in range(len(names_v)):
		system.name_positions_map[names_v[i]]=n+i
	system.names_v.extend(names_v)
	return names_d
			
def add_names(T,namesstr,category,d,names):
	a=[]
	for i in range(T):
		a.append(namesstr %(i,))
	names.extend(a)
	d[category]=a
	

def insert_arg(arg,add):
	n=min((len(arg),len(add)))
	arg[:n]=add[:n]
	return arg






