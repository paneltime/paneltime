#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains the argument class for the panel object
from ..output import stat_functions as stat
from .. import likelihood as logl

from .. import random_effects as re
from .. import functions as fu

from . import initreg

import numpy as np





class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,panel):
		p, q, d, k, m=panel.pqdkm
		self.initial_user_defined = False
		self.categories=['beta','rho','lambda','gamma','psi','omega']
		if panel.options.include_initvar:
			self.categories+=['initvar']

		if panel.z_active:
			self.categories+=['z']
		self.mu_removed=True
		if not self.mu_removed:
			self.categories+=['mu']
		self.make_namevector(panel,p, q, k, m)
		initargs=self.initargs(p, d, q, m, k, panel)
		self.position_defs(initargs)
		self.set_init_args(panel,initargs)
		self.get_user_constraints(panel)

	def initvar_asignment(self, initargs, omega, initvar, panel, rho , lmbda, beta, gamma, psi):
		
		p, q, d, k, m=panel.pqdkm

		if panel.options.include_initvar:
			initargs['initvar'][0][0] = initvar
		initargs['omega'][0][0] = omega
		initargs['beta']=beta

		for n,name, value in [(q, 'lambda', lmbda),
													(p, 'rho', rho),
													(k, 'gamma', gamma),
													(m, 'psi', psi),
													]:
			if n > 0:
				initargs[name][0][0] = value




	def get_user_constraints(self,panel):
		e="User contraints must be a dict of lists or a string evaluating to that, on the form of ll.args.dict_string."

		if type(panel.options.user_constraints)==dict:
			self.user_constraints=panel.options.user_constraints
		else:
			if panel.options.user_constraints is None or panel.options.user_constraints=='':
				self.user_constraints={}
				return
			try:
				self.user_constraints=eval(panel.options.user_constraints)
			except SyntaxError:
				print(f"Syntax error: {e}")
				self.user_constraints={}
				return			
			except Exception as e:
				print(e)
				self.user_constraints={}
				return
		if not panel.z_active and 'z' in self.user_constraints:
			self.user_constraints.pop('z')	
		if panel.options.include_initvar  and 'initvar' in self.user_constraints:
			self.user_constraints.pop('initvar')

		for grp in self.user_constraints:
			if not grp in self.caption_d:
				print(f"Constraint on {grp} not applied, {grp} not in arguments")
				self.user_constraints.pop(grp)



	def initargs(self,p,d,q,m,k,panel):

		args=dict()
		args['beta']=np.zeros((panel.X.shape[2],1))
		args['omega']=np.zeros((panel.W.shape[2],1))
		args['rho']=np.zeros((p,1))
		args['lambda']=np.zeros((q,1))
		args['psi']=np.zeros((m,1))
		args['gamma']=np.zeros((k,1))
		args['omega'][0][0]=0
		args['mu']=np.array([[]])
		if panel.options.include_initvar:
			args['initvar']=np.zeros((1,1))
		args['z']=np.array([[]])			

		if m>0 and panel.z_active:
			args['z']=np.array([[1e-09]])	

		if panel.N>1 and not self.mu_removed:
			args['mu']=np.array([[0.0001]])			


		return args

	def set_init_args(self,panel,initargs=None, default = True):
		p, q, d, k, m=panel.pqdkm

		if initargs is None:
			initargs = self.initargs(p, d, q, m, k, panel)

		#de2=np.roll(e**2,1)-e**2
		#c=stat.correl(np.concatenate((np.roll(de2,1),de2),2),panel)[0,1]
		beta,omega = self.set_init_regression(initargs,panel, default)
		self.init_var = omega
		self.args_start=self.create_args(initargs,panel)
		self.args_init=self.create_args(initargs,panel)
		self.set_restricted_args(p, d, q, m, k,panel,omega,beta)
		self.n_args=len(self.args_init.args_v)


	def set_restricted_args(self,p, d, q, m, k, panel,omega,beta):
		args_restricted=self.initargs(p, d, q, m, k, panel)
		args_OLS=self.initargs(p, d, q, m, k, panel)	
		
		args_restricted['beta'][0][0]=np.mean(panel.Y)
		args_OLS['beta']=beta
		
		v = panel.var(panel.Y)
		args_restricted['omega'][0][0]= panel.h_func(0, v, v)
		args_OLS['omega'][0][0]= panel.h_func(0,omega, omega)
			
		self.args_restricted=self.create_args(args_restricted,panel)
		self.args_OLS=self.create_args(args_OLS,panel)


	def create_null_ll(self,panel):
		if not hasattr(self,'LL_OLS'):
			self.LL_OLS=logl.LL(self.args_OLS,panel).LL
			self.LL_null=logl.LL(self.args_restricted,panel).LL	

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
			d[i]=np.array(args[rng]).reshape((n,1))
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
			if len(s)>0:
				v=np.concatenate((v,s))
		return v


	def make_namevector(self,panel,p, q, k, m):
		"""Creates a vector of the names of all regression varaibles, 
		including variables, ARIMA and GARCH terms. This defines the positions
		of the variables througout the estimation."""
		d, names_d = {}, {}
		captions=list(panel.input.X.keys())#copy variable names
		d['beta']=list(captions)
		c=[list(captions)]
		names = panel.input.X_names
		#names = [f'x{i}' for i in range(panel.n_beta)]
		names_d['beta'] = list(names)
		add_names(p,'rho%s    AR    p','rho',d,c,captions, names, names_d)
		add_names(q,'lambda%s MA    q','lambda',d,c,captions, names, names_d)
		add_names(k,'gamma%s  GARCH k','gamma',d,c,captions, names, names_d)
		add_names(m,'psi%s    ARCH  m','psi',d,c,captions, names, names_d)


		d['omega']=list(panel.input.W.keys())
		captions.extend(panel.input.W.keys())

		names_d['omega'] = [f'omega{i}' for i in range(panel.nW)]
		names.extend(names_d['omega'])

		c.append(d['omega'])
		if panel.options.include_initvar:
			d['initvar'] = ['Initial variance']
			captions.extend(d['initvar'])
			names_d['initvar'] = d['initvar']
			names.extend(names_d['initvar'])
			c.append(d['initvar'])
		
		if m>0:
			if panel.N>1 and not self.mu_removed:
				d['mu']=['mu (var.ID eff.)']
				captions.extend(d['mu'])
				names_d['mu']=['mu']
				names.extend(d['mu'])				
				c.append(d['mu'])

			if panel.z_active:
				d['z']=['z in h(e,z)']
				captions.extend(d['z'])
				names_d['z']=['z']
				names.extend(d['z'])					
				c.append(d['z'])

		self.caption_v=captions
		self.caption_d=d
		self.names_v = names
		self.names_d = names_d
		self.names_category_list=c

	def create_args(self,args,panel,constraints=None):
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
			captions=self.caption_d[c]
			a=args_d[c].flatten()
			for i in range(len(captions)):
				s.append(f"'{captions[i]}':{a[i]}")
			dict_string.append(f"'{c}':\n"+"{"+",\n".join(s)+"}")
		dict_string="{"+",\n".join(dict_string)+"}"
		return arguments_set(args_d, args_v, dict_string, self,panel)

	def test_consistency(self,args):
		#for debugging only
		m=self.positions_map
		for i in m:
			dict_arg=args.args_d[m[i][1]]
			if len(dict_arg.shape)==2:
				dict_arg=dict_arg[m[i][2]]
			if dict_arg[0]!=args.args_v[i]:
				raise RuntimeError("argument inconsistency")

	def get_name_ix(self,x,single_item=False):
		("Tests if x is recognized and returns the associated name and index."
   		"If not single_item, an array of indicies is returned")
		#returns name, list of indicies
		if x is None:
			return None, None
		if x in self.caption_v: 
			#if x is in individual caption names
			if single_item:
				indicies=self.caption_v.index(x)
			else:
				indicies=[self.caption_v.index(x)]	
			return x,indicies
		elif x in self.positions and not single_item:
			#if x is in group names
			if single_item:
				raise RuntimeError(f"'{x}' is a group name, single items do not exist for groups")
			indicies=list(self.positions[x])
			return x,indicies
		elif x in self.names_v:
			#if x is in individual names
			if single_item:
				indicies=self.names_v.index(x)
			else:
				indicies=[self.names_v.index(x)]	
			return x,indicies			
		try:
			#assuming x is an index
			name=self.caption_v[x]
		except Exception as e:
			raise RuntimeError(f"{e}. The identifier of an argument must be an integer or a string macthing a name in 'self.caption_v' or a category in 'self.positions'")
		
		# if no exeption occured, x is an integer, and we got its name
		
		if single_item:
			return name,x
		else:
			# Returns x as a list if multiple items are expected (single_item==False)
			return name,[x]

	def set_init_regression(self, initargs,panel, default):
		usrargs =  panel.options.arguments
		beta,rho,lmbda, psi, gamma,v, initvar, omega = initreg.start_values(panel)
		
		
		if not usrargs is None:#Checking for user arguments
			if type(usrargs)==str:
				try:
					usrargs = eval(usrargs.replace(" array"," np.array").replace(', dtype=float64',''))
				except NameError as e:
					if str(e)=="name 'array' is not defined":
						usrargs = eval(usrargs.replace("array"," np.array"))
			args = self.create_args(usrargs,panel)
			for c in args.args_d:
				initargs[c] = args.args_d[c]
			self.initial_user_defined = True
			
			
			return initargs['beta'], initargs['omega'][0,0]

		if panel.options.fixed_random_variance_eff==0:
			if v < 1e-20:
				print('Warning, your model may be over determined. Check that you do not have the dependent among the independents')	
				
		self.initvar_asignment(initargs, omega, initvar, panel, rho, lmbda, beta, gamma, psi)
		
		return beta, v
	
	def find_group(self, varname):
		return next((key for key, value in self.names_d.items() if varname in value), None)





def add_names(T,captionstr,category,d,c,captions, names, names_d):
	a=[]
	n=[]
	if ' ' in captionstr:
		namestr = captionstr.split(' ')[0]
	for i in range(T):
		a.append(captionstr %(i,))
		n.append(namestr %(i,))
	captions.extend(a)
	names.extend(n)
	d[category]=a
	names_d[category]=n
	c.append(a)


class arguments_set:
	"""A class that contains the numeric arguments used in the maximization
	in all shapes and forms needed."""
	def __init__(self,args_d,args_v,dict_string,arguments,panel):
		self.args_d=args_d#dictionary of arguments
		self.args_v=args_v#vector of arguments
		self.dict_string=dict_string#a string defining a dictionary of named arguments. For user input of initial arguments
		self.caption_v=arguments.caption_v#vector of captions
		self.caption_d=arguments.caption_d#dict of captions
		self.names_v=arguments.names_v#vector of names
		self.names_d=arguments.names_d#dict of names		
		self.n_args=len(self.args_v)
		self.pqdkm=panel.pqdkm
		self.positions=arguments.positions
		self.names_category_list=arguments.names_category_list
		self.create_args_names()

	def create_args_names(self):
		self.args_names = {}
		for a in self.args_d:
			for i, k in enumerate(self.names_d[a]):
				if k in self.args_names:
					raise RuntimeError(f'You have two instances of the name {k} in the regression. All names must be unique.')
				self.args_names[k] = self.args_d[a][i]
				try:
					self.args_names[k] = self.args_names[k][0]
				except IndexError as e:
					pass





