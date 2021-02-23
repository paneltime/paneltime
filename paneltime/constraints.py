#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import functions as fu
import calculus_functions as cf

MAX_COLLINEARITY=1e+7
EXTREEME_COLLINEARITY=1E+20
SMALL_COLLINEARITY=30

def add_static_constraints(constr,panel,ll,its):
	
	add_custom_constraints(constr,panel.args.user_constraints,ll)
	general_constraints=[('rho',-2,2),('lambda',-2,2),('gamma',-2,2),('psi',-2,2)]
	add_custom_constraints(constr,general_constraints,ll,False)
	p,q,d,k,m=panel.pqdkm
	if its==0 and len(panel.args.args_init.args_d['rho'])>0:
		if panel.args.args_init.args_d['rho'][0]==0:
			constr.add(panel.args.positions['rho'][0],None,'Initial MA constr',value=0.0)
	if panel.m_zero:
		constr.add(panel.args.positions['psi'][0],None,'GARCH input constr',value=0.05)
	sumsq_psi=0
	if ll is None:
		return
	for i in panel.args.positions['psi']:
		sumsq_psi+=ll.args.args_v[i]**2
	if sumsq_psi==0:
		for i in panel.args.positions['gamma']:
			constr.add(i,None,'GARCH term cannot be positive if ARCH terms are zero',value=0)
		

	
	
def add_dynamic_constraints(ll, direction):

	weak_mc_dict,CI,mc_problems,H_correl_problem=remove_all_multicoll(direction,ll)
	remove_singularities(direction.constr, direction.H)
	return weak_mc_dict,CI,mc_problems,H_correl_problem


	
def remove_singularities(constr,hessian):
	habs=np.abs(hessian)
	sing_problems=[]
	try:
		a=np.linalg.det(hessian)
		if a>0:
			return sing_problems
	except:
		for i in np.nonzero(np.diag(habs)>1e+100)[0]:
			constr.add(i,None,'singularity (extreeme value in diagonal)')
	for i in np.nonzero(np.diag(hessian)==0)[0]:
		constr.add(i,None,'singularity (zero in diagonal)')
		

def add_custom_constraints(constr,constraints,ll,override=True):
	"""Adds custom range constraints\n\n
		constraints shall be on the format [(name, minimum, maximum), ...]"""	
	if constraints is None or constraints=="":
		for c in constr.constraints:
			if constr[c].cause=='user constraint':
				constr.delete(c)
		return
	if type(constraints)==list:
		for c in constraints:
			add_custom_constraint_list(constr,c,ll,override)
	else:
		for grp in constraints:
			for name in constraints[grp]:
				add_custom_constraint_dict(constr,name,constraints[grp][name],ll,override)


def add_custom_constraint_list(constr,constraint,ll,override):
	"""Adds a custom range constraint\n\n
		constraint shall be on the format (name, minimum, maximum)"""
	name, minimum, maximum=constraint
	m=[minimum,maximum]
	for i in range(2):
		if type(m[i])==str:
			if not ll is None:
				m[i]=eval(m[i],globals(),ll.__dict__)
			else:
				return
	[minimum,maximum]=m
	constr.add_named(name,None,'user constraint', [minimum,maximum],override)


def add_custom_constraint_dict(constr,name,constraint,ll,override):
	"""Adds a custom range constraint\n\n
	   If list, constraint shall be on the format (minimum, maximum)"""
	if type(constraint)==list:
		constr.add_named(name,None,'user constraint', constraint,override)
	else:
		constr.add_named(name,None,'user constraint', [constraint,None],override)

class constraint:
	def __init__(self,index,assco,cause,value, interval,panel,category):
		names=panel.args.names_v
		self.name=names[index]
		self.intervalbound=None
		self.max=None
		self.min=None
		self.value=None
		self.value_str=None
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
		
class constraints:

	"""Stores the constraints of the LL maximization"""
	def __init__(self,panel,args):
		self.names=panel.args.names_v
		self.panel=panel
		self.categories={}
		self.fixed={}
		self.intervals={}
		self.constraints={}
		self.associates={}
		self.collinears={}
		self.args=args
		self.CI=None


	def add(self,index,assco,cause,interval=None,replace=True,value=None):
		"""Adds a constraint. 'index' is the position
		for which the constraints shall apply.  \n\n

		Equality constraints are chosen by specifying 'minimum_or_value' \n\n
		Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
		'replace' determines whether an existing constraint shall be replaced or not 
		(only one equality and inequality allowed per position)"""


		if ((not replace) and (index in self.constraints)):
			return False
		if interval is None:#this is a fixed constaint
			if value is None:
				value=self.args[index]
			if index in self.intervals:
				c=self.constraints[index]
				if not (c.min<value<c.max):
					return False
				else:
					self.intervals.pop(index)
		elif index in self.fixed: #this is an interval constraint
			self.fixed.pop(index)
			
		eq,category,j=self.panel.args.positions_map[index]
		if not category in self.categories:
			self.categories[category]=[index]
		elif not index in self.categories[category]:
			self.categories[category].append(index)

		c=constraint(index,assco,cause,value, interval ,self.panel,category)
		self.constraints[index]=c
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
	
	def reset_collinears(self,new_args):
		for i in self.collinears:
			self.constraints[i].value=new_args[i]
		
	def delete(self,index):
		if not index in self.constraints:
			return False
		self.constraints.pop(index)
		if index in self.intervals:
			self.intervals.pop(index)
		if index in self.fixed:
			self.fixed.pop(index)		
		eq,category,j=self.panel.args.positions_map[index]
		c=self.categories[category]
		if len(c)==1:
			if c[0]!=index:
				raise RuntimeError('this does not make sense')
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
		

	def add_named(self,name,assco,cause,interval,override=True):
		args=self.panel.args
		if name in args.names_v:
			indicies=[args.names_v.index(name)]		
		elif name in args.positions:
			indicies=[]
			positions=args.positions[name]
			for i in positions:
				indicies.append(i)
		else:
			return
		for i in indicies:
			if (not i in self.constraints) or override:
				if interval[1] is None:
					self.add(i,assco,cause, value=interval[0])
				else:
					self.add(i,assco,cause, interval)			
	
	def set_fixed(self,x):
		"""Sets all elements of x that has fixed constraints to the constraint value"""
		for i in self.fixed:
			x[i]=self.fixed[i].value
		
	def within(self,x,fix=False):
		"""Chekcs if x is within interval constraints. If fix=True, then elements of
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

def correl_IDs(p):
	IDs=[]
	appended=False
	x=np.array(p[:,1:3],dtype=int)
	for i,j in x:
		for k in range(len(IDs)):
			appended=append_to_ID(IDs[k],[i,j])
			if appended:
				break
		if not appended:
			IDs.append([i,j])
	g=len(IDs)
	keep=g*[True]
	for k in range(g):
		if keep[k]:
			for h in range(k+1,len(IDs)):
				appended=False
				for m in range(len(IDs[h])):
					if IDs[h][m] in IDs[k]:
						appended=append_to_ID(IDs[k],  IDs[h])
						keep[h]=False
						break
	g=[]
	for i in range(len(IDs)):
		if keep[i]:
			g.append(IDs[i])
	return g

def normalize(H,include):
	C=-H[include][:,include]
	d=np.maximum(np.diag(C).reshape((len(C),1)),1e-30)**0.5
	C=C/(d*d.T)
	includemap=np.arange(len(include))[include]
	return C,includemap
	
def decomposition(H,include=None):
	if include is None:
		include=[True]*len(H)
	C,includemap=normalize(H, include)
	c_index,var_prop=stat.var_decomposition(XXNorm=C)
	c_index=c_index.flatten()
	return c_index, var_prop,includemap
	
	
def multicoll_problems(direction,H,include,mc_problems):
	c_index, var_prop, includemap = decomposition(H, include)
	if c_index is None:
		return False,False
	mc_list=[]
	largest_ci=None
	for cix in range(1,len(c_index)):
		if ((np.sum(var_prop[-cix]>0.5)>1) or ((np.sum(var_prop[-cix]>0.3)>1) and (np.sum(var_prop[-cix]>0.99)>0))) and (c_index[-cix]>EXTREEME_COLLINEARITY):
			if largest_ci is None:
				largest_ci=c_index[-cix]
			if c_index[-cix]>SMALL_COLLINEARITY:
				var_prop_ix=np.argsort(var_prop[-cix])[::-1]
				var_prop_val=var_prop[-cix][var_prop_ix]
				j=var_prop_ix[0]
				j=includemap[j]
				done=var_prop_check(direction.panel,var_prop_ix, var_prop_val, includemap,j,mc_problems,c_index[-cix],mc_list)
				if done:
					break
	if len(mc_list)==0:
		return  c_index[-1],mc_list
	return c_index[-1],mc_list

def var_prop_check(panel,var_prop_ix,var_prop_val,includemap,assc,mc_problems,cond_index,mc_list):
	if cond_index>EXTREEME_COLLINEARITY:
		lim=0.3
	else:
		lim=0.5
	for i in range(1,len(var_prop_ix)):
		if var_prop_val[i]<lim:
			return True
		index=var_prop_ix[i]
		index=includemap[index]
		mc_problems.append([index,assc,cond_index])
		mc_list.append(index)
		return False
		
def add_mc_constraint(direction,mc_problems,weak_mc_dict):
	"""Adds constraints for severe MC problems"""
	constr=direction.constr
	if len(mc_problems)==0:
		return
	no_check=get_no_check(direction)
	a=[i[0] for i in mc_problems]
	if no_check in a:
		mc=mc_problems[a.index(no_check)]
		mc[0],mc[1]=mc[1],mc[0]
	for index,assc,cond_index in mc_problems:
		if (not index in weak_mc_dict) and (cond_index>SMALL_COLLINEARITY) and (cond_index<=MAX_COLLINEARITY) and (not index==no_check):
			weak_mc_dict[index]=[assc,cond_index]#contains also collinear variables that are only slightly collinear, which shall be restricted when calcuating CV-matrix.	
		if not ((index in constr.associates) or (index in constr.collinears)) and cond_index>MAX_COLLINEARITY and (not index==no_check):
			constr.add(index,assc,'collinear')
		
def get_no_check(direction):
	no_check=direction.panel.settings.do_not_constrain.value
	x_names=direction.panel.input.x_names
	if not no_check is None:
		if no_check in x_names:
			return x_names.index(no_check)
		print("A variable was set for the 'Do not constraint' option (do_not_constrain), but it is not among the x-variables")
			
	
		
def remove_all_multicoll(direction,ll):
	k,k=direction.H.shape
	weak_mc_dict=dict()
	include=np.array(k*[True])
	include[list(direction.constr.fixed)]=False
	H_correl_problem=constraint_correl_cluster(direction,include)
	mc_problems=[]#list of [index,associate,condition index]
	CI_max=constraint_multicoll(k, direction, include, mc_problems)
	add_mc_constraint(direction,mc_problems,weak_mc_dict)
	select_arma(direction.constr, ll)
	return weak_mc_dict,CI_max,mc_problems,H_correl_problem

def constraint_multicoll(k,direction,include,mc_problems):
	CI_max=0
	for i in range(k-1):
		CI,mc_list=multicoll_problems(direction,direction.H,include,mc_problems)
		CI_max=max((CI_max,CI))
		if len(mc_list)==0:
			break
		include[mc_list]=False
	return CI_max
		
def constraint_correl_cluster(direction,include):
	if int(direction.its/2)==direction.its/2 and direction.its>0:
		return False		
	H=direction.H
	dH=np.diag(H).reshape((len(H),1))
	corr=np.abs(H/(np.abs(dH*dH.T)+1e-100)**0.5)
	np.fill_diagonal(corr,0)
	problems=list(np.unique(np.nonzero((corr>1-1e-12)*(corr<1))[0]))
	if len(problems)==0:
		return False
	#not_problem=np.argsort(dH[problems].flatten())[0]
	#problems.pop(not_problem)
	allix=list(range(len(H)))
	for i in problems:
		allix.pop(allix.index(i))
	if len(allix)==0:
		return	False
	not_problem=np.argsort(dH[allix].flatten())[-1]
	allix.pop(not_problem)	
	for i in allix:
		include[i]=False
		if not i in direction.constr.fixed:
			direction.constr.add(i,None,'Perfect correlation in hessian')
	return True
		
	
	

def select_arma(constr,ll):
	for i in constr.collinears:
		if constr.constraints[i].category in ['rho','lambda', 'gamma','psi']:
			assc=constr.collinears[i]
			if not assc in constr.fixed:
				if abs(ll.args.args_v[assc])<abs(ll.args.args_v[i]):
					constr.delete(i)
					constr.add(assc,i,'collinear')
	
	
			
		


