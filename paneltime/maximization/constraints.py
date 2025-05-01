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
	def __init__(self,panel,args, its, armaconstr):
		dict.__init__(self)
		self.categories={}
		self.mc_constr = {}
		self.mc_report = {}
		self.fixed={}
		self.intervals={}
		self.associates={}
		self.collinears={}
		self.mc_report={}
		self.args=args
		self.args[0]
		self.panel_args=panel.args
		self.ci=None
		self.its=its
		self.initvar_set = False
		self.pqdkm=panel.pqdkm
		self.m_zero=panel.m_zero
		self.ARMA_constraint = armaconstr
		self.H_correl_problem=False
		self.is_collinear = False
		self.constr_matrix_beta = [
				 (1, 0, 0, 0, 1), 
				 (0, 1, 0, 0, 1),
				 (0, 0, 1, 0, 1), 
				 (0, 0, 0, 1, 1)
		]
		self.constr_matrix = [
				 (1, 0, 0, 0, 0), 
				 (0, 1, 0, 0, 0),
				 (0, 0, 1, 0, 0), 
				 (0, 0, 0, 1, 0)
              
				]  

		#self.constr_matrix = []


	def add(self,name,assco,cause,interval=None,replace=True,value=None, ci = 0):
		#(self,index,assco,cause,interval=None,replace=True,value=None)
		name,index=self.panel_args.get_name_ix(name)
		name_assco,assco=self.panel_args.get_name_ix(assco,True)
		for i in index:
			self.add_item(i,assco,cause, interval ,replace,value, ci)

	def clear(self,cause=None):
		for c in list(self.keys()):
			if self[c].cause==cause or cause is None:
				self.delete(c)	

	def add_item(self,index,assco,cause,interval,replace,value, ci):
		"""Adds a constraint. 'index' is the position
		for which the constraints shall apply.  \n\n

		Equality constraints are chosen by specifying 'minimum_or_value' \n\n
		Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
		'replace' determines whether an existing constraint shall be replaced or not 
		(only one equality and inequality allowed per position)"""

		args=self.panel_args
		if not replace:
			if index in self:
				return False

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

		c = Constraint(index,assco,cause,value, interval ,args.caption_v,category, ci)
		self[index] = c
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

	def add_static_constraints(self, comput, its, ll= None):

		panel = comput.panel
		pargs=self.panel_args
		p, q, d, k, m=self.pqdkm

		if its<-4:
			c = 0.5
		else:
			c=self.ARMA_constraint


		constraints=[('rho',-c,c),('lambda',-c,c),('gamma',-c,c),('psi',-c,c)]
		if panel.options.include_initvar:
			constraints.append(('initvar',1e-50,1e+10))
		for name, min_, max_ in constraints:
				self.add(name,None,'ARMA/GARCH extreme bounds', [min_,max_])
		self.add_custom_constraints(panel, pargs.user_constraints, True, 'user constraints')
	
		c = self.constr_matrix
		if its<len(c):
			constr = self.get_init_constr(*c[its])
			for name in constr:
				self.add(name, None,'user constraint')

		if not comput.constr_old is None:
			if not comput.constr_old.ci is None:
				if comput.constr_old.ci > 15 or self.initvar_set:
					self.initvar_set = True
					self.add('initvar', None,'initial variance set')
		a=0
		
			
			
	def get_init_constr(self, p0,q0,k0,m0, beta):
		p, q, d, k, m = self.pqdkm
		constr_list = ([f'rho{i}' for i in range(p0,p)] +
									 [f'lambda{i}' for i in range(q0,q)] + 
									 [f'gamma{i}' for i in range(k0,k)] +
									 [f'psi{i}' for i in range(m0,m)])
		if beta>0:
			constr_list.append(('beta',None))
		return constr_list
		
		

	def add_dynamic_constraints(self,computation, H, ll, args = None):
		if not args is None:
			self.args = args
			self.args[0]
		k,k=H.shape
		incl=np.array(k*[True])
		incl[list(computation.constr.fixed)]=False
		self.constraint_multicoll(k, computation, incl, H)



	def constraint_multicoll(self, k,computation,incl, H):
		cimax = {}
		for i in range(k-1):
			ci = self.multicoll_problems(computation, H, incl, cimax)
			if len(self.mc_report)==0:
				break
			incl[list(self.mc_constr)]=False
		if len(cimax)==0:
			self.ci, self.ci_n = None, None
		else:
			self.ci = max(cimax, key = cimax.get)
			self.ci_n = cimax[self.ci]

	def multicoll_problems(self, computation, H, incl, cimax):
		c_index, var_prop, includemap, d, C = decomposition(H, incl)
		if any(d==0):
			self.remove_zero_eigenvalues(C, incl, includemap, d)
			c_index, var_prop, includemap, d, C = decomposition(H, incl)

		if c_index is None:
			return 0
		limit_report = computation.panel.options.multicoll_threshold_report
		limit_constr = computation.multicoll_threshold_max

		for cix in range(1,len(c_index)):
	
			constr = self.add_collinear(limit_constr, self.mc_constr, c_index[-cix], 
					   						True, var_prop[-cix], includemap, cimax)
			
			report = self.add_collinear(limit_report, self.mc_report, c_index[-cix], 
					   						True, var_prop[-cix], includemap, cimax)
			if constr:
				break
		return c_index[-1]

	def add_collinear(self, limit, ci_list, ci, constrain, var_dist, includemap, cimax):
		"""Identifies if there are collinearities outside `limit`, and adds them to ci_list. If `constrain==True`, 
		a regression constraint is added."""
		sign_var = var_dist > 0.5


		if (not np.sum(sign_var)>1) or ci<limit:
			return False
		a = np.argsort(var_dist)
		index = includemap[a[-1]]
		assc = includemap[a[-2]]
		if ci in cimax:
			cimax[ci] = max((np.sum(sign_var), cimax[ci]))
		else:
			cimax[ci] = np.sum(sign_var)
		ci_list[index] = assc
		if constrain:
			#print(f"{index}/{m}")
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


	def add_custom_constraints(self, panel, constraints,replace,cause):
		"""Adds a custom range constraint\n\n
			 If list, constraint shall be on the format (minimum, maximum)"""
		#If it is a dict, it needs to have a paneltime hierarchical structure with
		#grops at top level and 
		for grp in constraints:
			c=constraints[grp]
			if c is None:
				continue
			elif type(c) == tuple: #interval constraints on whole group
				self.add(grp,None,cause, c,replace)
			elif type(c) == float:
				self.add(grp,None,cause, replace = replace, value = c)
			else:
				for i, name in enumerate(panel.args.caption_d[grp]):
					self.add_custom_constraint_subgroup(c, i, name, replace, cause, grp)

	def add_custom_constraint_subgroup(self, constraints, i, name, replace, cause, grp):
		c = constraints
		if not len(c)>i:
			return
		if c[i] is None:
			return
		if type(c[i]) == tuple:
			self.add(name,None,cause, c[i],replace)
		elif type(c[i]) == float:
			self.add(name,None,cause, value = c[i], replace = replace)
		elif type(c[i][0]) == float and len(c[i])==1:
			self.add(name,None,cause, value = c[i][0], replace = replace)
		else:
			raise RuntimeError(f"When using the constraints option, the elements of {grp} "
					  			"must either be a tuple with (max, min), a float or a single element list with a float "
									)
					
	def __str__(self):
		s = ''
		for desc, obj in [('All', self),
								('Fixed', self.fixed),
								('Intervals', self.intervals)]:
			s += f"{desc} constraints:\n"
			for i in obj:
				c=obj[i]
				try:
					s += f"constraint: {i}, associate:{c.assco_ix}, max:{c.max}, min:{c.min}, value:{c.value}, cause:{c.cause}\n"
				except:
					s += f"constraint: {i}, associate:{c.assco_ix}, max:{None}, min:{None}, value:{c.value}, cause:{c.cause}\n"  
		return s
	
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








