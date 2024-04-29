#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from itertools import combinations
path = os.path.dirname(__file__)
from ..output import stat_functions as stat


import numpy as np



class Constraint:
	def __init__(self,index,assco,cause,value, interval,names,category, ci):
		self.name=names[index]
		self.intervalbound=None
		self.max=None
		self.min=None
		self.value=None
		self.value_str=None
		self.ci = ci
		if interval is None:
			self.value=value
			self.value_str=str(round(self.value,8))
		else:
			if interval[0]>interval[1]:
				raise RuntimeError('Lower constraint cannot exceed upper')
			self.min=interval[0]
			self.max=interval[1]
			self.cause='user/general constraint'
		self.assco_ix=assco
		if assco is None:
			self.assco_name=None
		else:
			self.assco_name=names[assco]
		self.cause=cause
		self.category=category	

class Constraints(dict):

	"""Stores the constraints of the LL maximization"""	
	def __init__(self,panel,args, its, armaconstr, betaconstr):
		dict.__init__(self)
		self.categories={}
		self.mc_list = set()
		self.mc_problems = set()
		self.initvar_fixed = False
		self.fixed={}
		self.intervals={}
		self.associates={}
		self.collinears={}
		self.weak_mc_dict={}
		self.args=args
		self.args[0]
		self.panel_args=panel.args
		self.CI=None
		self.its=its
		self.pqdkm=panel.pqdkm
		self.m_zero=panel.m_zero
		self.ARMA_constraint = armaconstr
		self.H_correl_problem=False
		self.is_collinear = False
		self.constr_matrix_beta = [
				 (0, 0, 0, 0, 1), 
				 (1, 0, 0, 0, 1), 
				 (0, 1, 0, 0, 1),
				 (1, 1, 0, 0, 1), 
				 (0, 0, 1, 0, 1),
				 (0, 0, 0, 1, 1), 
				 (0, 0, 1, 1, 1), 
				 (1, 1, 1, 1, 1)
		]
		self.constr_matrix = [
				 (0, 0, 0, 0, 0), 
				 (1, 0, 0, 0, 0), 
				 (0, 1, 0, 0, 0),
				 (1, 1, 0, 0, 0), 
				 (0, 0, 1, 0, 0),
				 (0, 0, 0, 1, 0), 
				 (0, 0, 1, 1, 0), 
				 (1, 1, 1, 1, 0),                
				]  
		if betaconstr: 
			self.constr_matrix = self.constr_matrix_beta + self.constr_matrix

	def add(self,name,assco,cause,interval=None,replace=True,value=None,args=None, ci = 0):
		#(self,index,assco,cause,interval=None,replace=True,value=None)
		name,index=self.panel_args.get_name_ix(name)
		name_assco,assco=self.panel_args.get_name_ix(assco,True)
		for i in index:
			self.add_item(i,assco,cause, interval ,replace,value,args, ci)

	def clear(self,cause=None):
		for c in list(self.keys()):
			if self[c].cause==cause or cause is None:
				self.delete(c)	

	def add_item(self,index,assco,cause,interval,replace,value,args, ci):
		"""Adds a constraint. 'index' is the position
		for which the constraints shall apply.  \n\n

		Equality constraints are chosen by specifying 'minimum_or_value' \n\n
		Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
		'replace' determines whether an existing constraint shall be replaced or not 
		(only one equality and inequality allowed per position)"""

		if args is None:
			args=self.panel_args
		if not replace:
			if index in self:
				return False
		interval,value=test_interval(interval, value)
		if interval is None:#this is a fixed constraint
			if len(self.fixed)==len(args.caption_v)-1:#can't lock all variables
				return False
			if value is None:
				value=self.args[index]
			if index in self.intervals:
				c=self[index]
				if not (c.min<value<c.max):
					return False
				else:
					self.intervals.pop(index)
		elif index in self.fixed: #this is an interval constraint, no longer a fixed constraint
			self.fixed.pop(index)

		eq,category,j=self.panel_args.positions_map[index]
		if not category in self.categories:
			self.categories[category]=[index]
		elif not index in self.categories[category]:
			self.categories[category].append(index)

		c=Constraint(index,assco,cause,value, interval ,args.caption_v,category, ci)
		self[index]=c
		if value is None:
			self.intervals[index]=c
		else:
			self.fixed[index]=c
		if not assco is None:
			if not assco in self.associates:
				self.associates[assco]=[index]
			elif not index in self.associates[assco]:
				self.associates[assco].append(index)
		if cause=='collinear':
			self.collinears[index]=assco
		return True

	def delete(self,index):
		if not index in self:
			return False
		self.pop(index)
		if index in self.intervals:
			self.intervals.pop(index)
		if index in self.fixed:
			self.fixed.pop(index)		
		eq,category,j=self.panel_args.positions_map[index]
		c=self.categories[category]
		if len(c)==1:
			self.categories.pop(category)
		else:
			i=np.nonzero(np.array(c)==index)[0][0]
			c.pop(i)
		a=self.associates
		for i in a:
			if index in a[i]:
				if len(a[i])==1:
					a.pop(i)
					break
				else:
					j=np.nonzero(np.array(a[i])==index)[0][0]
					a[i].pop(j)
		if index in self.collinears:
			self.collinears.pop(index)
		return True


	def set_fixed(self,x):
		"""Sets all elements of x that has fixed constraints to the constraint value"""
		for i in self.fixed:
			x[i]=self.fixed[i].value

	def within(self,x,fix=False):
		"""Checks if x is within interval constraints. If fix=True, then elements of
		x outside constraints are set to the nearest constraint. if fix=False, the function 
		returns False if x is within constraints and True otherwise"""
		for i in self.intervals:
			c=self.intervals[i]
			if (c.min<=x[i]<=c.max):
				c.intervalbound=None
			else:
				if fix:
					x[i]=max((min((x[i],c.max)),c.min))
					c.intervalbound=str(round(x[i],8))
				else:
					return False
		return True

	def add_static_constraints(self, panel, its, ll=None, constr = []):
		pargs=self.panel_args
		p, q, d, k, m=self.pqdkm

		if its<-4:
			c = 0.5
		else:
			c=self.ARMA_constraint


		general_constraints=[('rho',-c,c),('lambda',-c,c),('gamma',0,1),('psi',0,1)]
		if panel.options.include_initvar.value:
			general_constraints.append(('initvar',1e-50,1e+50))
		self.add_custom_constraints(panel, general_constraints, ll)
		self.add_custom_constraints(panel, pargs.user_constraints, ll)
		self.add_custom_constraints(panel, constr, ll)
		

		c = self.constr_matrix
		if its<len(c):
			constr = self.get_init_constr(*c[its])
			self.add_custom_constraints(panel, constr, ll)
		a=0
		
			
			
	def get_init_constr(self, p0,q0,k0,m0, beta):
		p, q, d, k, m = self.pqdkm
		constr_list = ([(f'rho{i}', None) for i in range(p0,p)] +
									 [(f'lambda{i}', None) for i in range(q0,q)] + 
									 [(f'gamma{i}', None) for i in range(k0,k)] +
									 [(f'psi{i}', None) for i in range(m0,m)])
		if beta>0:
			constr_list.append(('beta',0))
		return constr_list
		
		

	def add_dynamic_constraints(self,computation, H, ll, args = None):
		if not args is None:
			self.args = args
			self.args[0]
		k,k=H.shape
		self.weak_mc_dict=dict()
		incl=np.array(k*[True])
		incl[list(computation.constr.fixed)]=False
		self.CI = self.constraint_multicoll(k, computation, incl, H)



	def constraint_multicoll(self, k,computation,incl, H):
		CI_max=0
		for i in range(k-1):
			CI = self.multicoll_problems(computation, H, incl)
			CI_max=max((CI_max,CI))
			if len(self.mc_list)==0:
				break
			incl[list(self.mc_list)]=False
		return CI_max

	def multicoll_problems(self, computation, H, incl):
		c_index, var_prop, includemap, d, C = decomposition(H, incl)
		if any(d==0):
			self.remove_zero_eigenvalues(C, incl, includemap, d)
			c_index, var_prop, includemap, d, C = decomposition(H, incl)

		if c_index is None:
			return False,[]
		limit_report = computation.panel.options.multicoll_threshold_report.value
		limit = computation.panel.options.multicoll_threshold_max.value

		for cix in range(1,len(c_index)):
			for lmt,lst in [(limit, self.mc_list), 
						(limit_report, self.mc_problems)]:
				added = self.add_collinear(lmt, lst, c_index[-cix], 
					   lmt==limit, var_prop[-cix], includemap, computation)
				if added:
					break
		return c_index[-1]

	def add_collinear(self, limit, ci_list, ci, constrain, var_dist, includemap, computation):
		sign_var = var_dist > 0.5

		n = len(sign_var)
		m = len(self.panel_args.names_v)
		if self.initvar_fixed == True:
			self.add(m-1 ,None,'initvar restr')
		if (not np.sum(sign_var)>1) or ci<limit:
			return False
		a = np.argsort(var_dist)
		index = includemap[a[-1]]
		assc = includemap[a[-2]]
		if index == m-2:
			index = includemap[a[-2]]
			assc = includemap[a[-1]]	
			#self.initvar_fixed = True	
		ci_list.add(index)
		if constrain:
			print(f"{index}/{m}")
			self.add(index ,assc,'collinear', ci = ci)
			return True
		return False

	def remove_zero_eigenvalues(self, C, incl, includemap, d):
		combo =  find_singular_combinations(C, d)
		if combo is None:
			return
		for i in combo:
			indx = includemap[i]
			incl[indx] = False
			self.add(indx, None,'zero ev')

	def add_custom_constraints(self,panel, constraints, ll,replace=True,cause='user constraint',clear=False,args=None):
		"""Adds custom range constraints\n\n
			constraints shall be on the format [(name, minimum, maximum), ...]
			or
			a dictionary of the same form of an argument dictionary
			(a dictionary of dictonaries on the form dict[category][name]) 
			where items are lists of [minimum,maximum] or the fixing value

			name is taken from self.panel_args.caption_v"""	

		if clear:
			self.clear(cause)
		if type(constraints)==list or type(constraints)==tuple:
			for c in constraints:
				self.add_custom_constraints_list(c,replace,cause,args, ll)
		elif type(constraints)==dict:
			self.add_custom_constraints_dict(panel, constraints,replace,cause,args)
		else:
			raise TypeError("The constraints must be a list, tuple or dict.")


	def add_custom_constraints_list(self,constraint,replace,cause,args, ll):
		"""Adds a custom range constraint\n\n
			constraint shall be on the format (name, minimum, maximum)"""
		if np.issubdtype(type(constraint), np.integer):
			name = self.panel_args.names_v[constraint]
			self.add(name, None,cause,replace=replace,args=args)
			return
		elif type(constraint)==str or isinstance(constraint,int):
			name, value=constraint,None
			self.add(name,None,cause,replace=replace,value=value,args=args)
			return
		elif len(constraint)==2:
			name, minimum,maximum=list(constraint)+[None]
		elif len(constraint)==3:
			name, minimum, maximum=constraint
		else:
			raise TypeError("A constraint needs to be a string, integer or iterable with lenght no more than tree.")

		name,ix=self.panel_args.get_name_ix(name)
		m=[minimum,maximum]
		for i in range(2):
			if type(m[i])==str:
				try:
					m[i]=eval(m[i],globals(),ll.__dict__)
				except:
					print(f"Custom constraint {name} ({m[i]}) failed")
					return
		[minimum,maximum]=m
		self.add(name,None,cause, [minimum,maximum],replace,args=args)

	def add_custom_constraints_dict(self, panel, constraints,replace,cause,args):
		"""Adds a custom range constraint\n\n
			 If list, constraint shall be on the format (minimum, maximum)"""
		for grp in constraints:
			for i in range(len(panel.args.caption_d[grp])):
				name = panel.args.caption_d[grp][i]
				c=constraints[grp]
				try:
					iter(c)
					self.add(name,None,cause, c,replace,args=args)
				except TypeError as e:
					self.add(name,None,cause, [c,None],replace,args=args)



					
	def print(self, kind = None):
		for desc, obj in [('All', self),
								('Fixed', self.fixed),
								('Intervals', self.intervals)]:
			print(f"{desc} constraints:")
			for i in obj:
				c=obj[i]
				try:
					print(f"constraint: {i}, associate:{c.assco_ix}, max:{c.max}, min:{c.min}, value:{c.value}, cause:{c.cause}")  
				except:
					print(f"constraint: {i}, associate:{c.assco_ix}, max:{None}, min:{None}, value:{c.value}, cause:{c.cause}")  

def test_interval(interval,value):
	if not interval is None:
		if np.any([i is None for i in interval]):
			if interval[0] is None:
				value=interval[1]
			else:
				value=interval[0]
			interval=None	
	return interval,value

def append_to_ID(ID,intlist):
	inID=False
	for i in intlist:
		if i in ID:
			inID=True
			break
	if inID:
		for j in intlist:
			if not j in ID:
				ID.append(j)
		return True
	else:
		return False

def normalize(H,incl):
	C=-H[incl][:,incl]
	d=np.maximum(np.diag(C).reshape((len(C),1)),1e-30)**0.5
	C=C/(d*d.T)
	includemap=np.arange(len(incl))[incl]
	return C,includemap

def decomposition(H,incl=None):
	C,includemap=normalize(H, incl)
	c_index, var_prop, d, p = stat.var_decomposition(xx_norm = C)
	if any(d==0):
		return None, None,includemap, d, C
	c_index=c_index.flatten()
	return c_index, var_prop,includemap, d, C


def find_singular_combinations(matrix, evs):
	n = matrix.shape[0]
	rank = len(matrix)-sum(evs==0)
		
	if rank == n:
		print("Matrix is not singular.")
		return None
	
	# Find all combinations of columns that might be causing singularity
	for i in range(1, n - rank + 1):  # Adjust based on how many you need to remove
		for combo in combinations(range(n), i):
			reduced_matrix = np.delete(matrix, combo, axis=1)
			reduced_matrix = np.delete(reduced_matrix, combo, axis=0)  # Remove corresponding rows

			if sum(np.linalg.eigvals(reduced_matrix)==0) == 0:
				return combo  # Found the combination causing singularity
		
	return None  # In case no combination found, though this should not happen








