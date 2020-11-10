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
		initargs=self.initargs(p, d, q, m, k, panel)
		self.position_defs(initargs)
		self.set_init_args(initargs)
		self.init_args(panel)
		
		
	def init_args(self,panel):
		e="User contraints must be a dict of dicts or a string evaluating to that, on the form of ll.args.dict_string. User constraints not applied"
		if panel.settings.user_constraints.value is None or panel.settings.user_constraints.value=='':
			self.user_constraints={}
			return
		try:
			self.user_constraints=eval(panel.settings.user_constraints.value)
		except SyntaxError:
			print(f"Syntax error: {e}")
		except:
			print(e)
			self.user_constraints={}
			return
		if not panel.z_active and 'z' in self.user_constraints:
			self.user_constraints.pop('z')	
		for i in self.user_constraints:
			for j in self.user_constraints[i]:
				if j not in self.names_v:
					print(f"Constraint on {name} not applied (name not found in arguments)")
		

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

	def set_init_args(self, initargs,default=False):
		panel=self.panel
		p, q, d, k, m=panel.pqdkm

		#de2=np.roll(e**2,1)-e**2
		#c=stat.correl(np.concatenate((np.roll(de2,1),de2),2),panel)[0,1]
		
		beta,e=stat.OLS(panel,panel.X,panel.Y,return_e=True)
		#beta=stat.OLS_simple(panel.input.Y,panel.input.X,True,False)
		self.init_e_st=e[panel.included]
		self.init_e_st=self.init_e_st/np.var(self.init_e_st)**0.5
		initargs['beta']=beta
		if panel.settings.variance_fixed_random_eff.value==0:
			initargs['omega'][0]=np.log(panel.var(e))
			if panel.var(e)<1e-20:
				print('Warning, your model may be over determined. Check that you do not have the dependent among the independents')
		self.args_start=self.create_args(initargs)
		if not default:
			#previous arguments
			self.process_init_user_args(self.panel.input.args_archive.args,
												initargs,'loaded')
			#user defined arguments:
			self.process_init_user_args(self.panel.input.args,
												initargs,'user defined')
		self.args_init=self.create_args(initargs)
		self.set_restricted_args(p, d, q, m, k,panel,e,beta)
		self.n_args=len(self.args_init.args_v)
		
	def process_init_user_args(self,old_args,initargs,errstr):
		if old_args is None:
			return
		oargs=old_args
		if isinstance(old_args,arguments_set):
			oargs=eval(old_args.dict_string)
		elif type(old_args)==str:
			oargs=eval(old_args)
		if type(oargs)!=dict:
			raise RuntimeError(f"The {errstr} arguments need to be a dictionary, not {type(oargs)}")
		for cat in oargs:
			for name in oargs[cat]:
				if name in self.names_d[cat]:
					k=self.names_d[cat].index(name)
					initargs[cat][k]=oargs[cat][name]

		

	def set_restricted_args(self,p, d, q, m, k, panel,e,beta):
		args_restricted=self.initargs(p, d, q, m, k, panel)
		args_restricted['beta'][0][0]=np.mean(panel.Y)
		args_restricted['omega'][0][0]=np.log(panel.var(panel.Y))
		self.args_restricted=self.create_args(args_restricted)
		
		args_OLS=self.initargs(p, d, q, m, k, panel)	
		args_OLS['beta']=beta
		args_OLS['omega'][0][0]=np.log(panel.var(e))
		self.args_OLS=self.create_args(args_OLS)
		
	
	def create_null_ll(self):
		if not hasattr(self,'LL_OLS'):
			self.LL_OLS=logl.LL(self.args_OLS,self.panel).LL
			self.LL_null=logl.LL(self.args_restricted,self.panel).LL	
		
	def position_defs(self,initargs):
		"""Defines positions in vector argument"""

		self.positions=dict()
		self.positions_map=dict()#a dictionary of indicies containing the string name and sub-position of index within the category
		k=0
		for i in self.categories:
			n=len(initargs[i])
			rng=range(k,k+n)
			self.positions[i]=rng
			for j in rng:
				self.positions_map[j]=[0,i,j-k]#equation,category,relative position
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
			d[i]=np.array(args[rng])
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
			s=np.array(args[i])
			if len(s.shape)==2:
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
		c=[list(names)]
		add_names(p,'rho%s    AR    p','rho',d,c,names)
		add_names(q,'lambda%s MA    q','lambda',d,c,names)
		add_names(k,'gamma%s  GARCH k','gamma',d,c,names)
		add_names(m,'psi%s    ARCH  m','psi',d,c,names)
		
		
		d['omega']=panel.input.W_names
		c.append(d['omega'])
		
		names.extend(panel.input.W_names)
		if m>0:
			if panel.N>1 and not self.mu_removed:
				d['mu']=['mu (var.ID eff.)']
				names.extend(d['mu'])
				c.append(d['mu'])
			if panel.z_active:
				d['z']=['z in h(e,z)']
				names.extend(d['z'])
				c.append(d['z'])
			
		self.names_v=names
		self.names_d=d
		self.names_category_list=c
		
	def insert_arg(self,argname,args,AR_type=False):
		
		if self.panel.input.args_archive.args is None:
			return
		arg=args[argname]
		arg_old=self.panel.input.args_archive.args[argname]
		names=self.names_d[argname]
		names_old=self.panel.input.args_archive.args.names_d[argname]
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
			
	def create_args(self,args,constraints=None):
		if isinstance(args,arguments_set):
			self.test_consistency(args)
			return args
		args_v=self.conv_to_vector(args)
		if not constraints is None:
			constraints.within(args_v,True)	
			constraints.set_fixed(args_v)
		args_d=self.conv_to_dict(args_v)
		dict_string=[]
		for c in self.categories:
			s=[]
			names=self.names_d[c]
			a=args_d[c].flatten()
			for i in range(len(names)):
				s.append(f"'{names[i]}':{a[i]}")
			dict_string.append(f"'{c}':\n"+"{"+",\n".join(s)+"}")
		dict_string="{"+",\n".join(dict_string)+"}"
		return arguments_set(args_d, args_v, dict_string, self)
	
	def test_consistency(self,args):
		#for debugging only
		m=self.positions_map
		for i in m:
			dict_arg=args.args_d[m[i][1]]
			if len(dict_arg.shape)==2:
				dict_arg=dict_arg[m[i][2]]
			if dict_arg[0]!=args.args_v[i]:
				raise RuntimeError("argument inconsistency")
			
		
			
def add_names(T,namsestr,category,d,c,names):
	a=[]
	for i in range(T):
		a.append(namsestr %(i,))
	names.extend(a)
	d[category]=a
	c.append(a)
	
	
class arguments_set:
	"""A class that contains the arguments in all shapes and forms needed."""
	def __init__(self,args_d,args_v,dict_string,arguments):
		self.args_d=args_d#dictionary of arguments
		self.args_v=args_v#vector of arguments
		self.dict_string=dict_string#a string defining a dictionary of named arguments. For user input of initial arguments
		self.names_v=arguments.names_v#vector of names
		self.names_d=arguments.names_d#dict of names
		self.n_args=len(self.args_v)
		self.pqdkm=arguments.panel.pqdkm
		self.positions=arguments.positions
		self.names_category_list=arguments.names_category_list
		