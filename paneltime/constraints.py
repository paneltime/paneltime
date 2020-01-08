#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import functions as fu

MAX_COLLINEARITY=1e+18
SMALL_COLLINEARITY=30

def add_static_constraints(constr,panel,ll,its):
	
	add_custom_constraints(constr,panel.settings.user_constraints.value,ll)
	general_constraints=[('rho',-2,2),('lambda',-2,2),('gamma',-2,2),('psi',-2,2)]
	#add_custom_constraints(constr,general_constraints,ll)
	p,q,d,k,m=panel.pqdkm
	
	if panel.m_zero:
		constr.add(panel.args.positions['psi'][0],None,'GARCH input constr',value=0.05)
	fix_gamma=(k*m>0)
	if not ll is None and fix_gamma:
		fix_gamma=ll.args_d['psi'][0]==0
	if fix_gamma and (not panel.settings.loadargs.value):
		for i in panel.args.positions['gamma']:
			constr.add(i,None,'GARCH term cannot be positive if ARCH term is zero',value=0)
		

	
	
def add_dynamic_constraints(ll, direction,newton_failed):

	mc_dict=remove_all_multicoll(direction,ll,newton_failed)
	remove_singularities(direction.constr, direction.H)
	return mc_dict


	
def remove_singularities(constr,hessian):
	n=len(hessian)
	habs=np.abs(hessian)
	m=np.max(habs)
	h_temp=np.array(hessian)
	try:
		a=np.linalg.det(h_temp)
	except:
		for i in np.nonzero(np.abs(np.diag(h_temp))>1e+100)[0]:
			constr.add(i,None,'singularity')

		

def add_custom_constraints(constr,constraints,ll):
	"""Adds custom range constraints\n\n
		constraints shall be on the format [(name, minimum, maximum), ...]"""	
	if constraints is None:
		return
	for i in constraints:
		add_custom_constraint(constr,i,ll)


def add_custom_constraint(constr,constraint,ll):
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
	constr.add_named(name,None,'user constraint', [minimum,maximum])



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
		

	def add_named(self,name,assco,cause,interval):
		positions=self.panel.args.positions[name]
		for i in positions:
			if interval[1] is None:
				self.add(i,assco,cause, value=interval[0])
			else:
				self.add(i,assco,cause, interval)
		return
	
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
	
def decomposition(H,include):
	C,includemap=normalize(H, include)
	c_index,var_prop=stat.var_decomposition(XXNorm=C)
	try:
		c_index,var_prop=stat.var_decomposition(XXNorm=C)
	except:
		return None,None,None
	c_index=c_index.flatten()
	return c_index, var_prop,includemap
	
	
def multicoll_problems(direction):
	constr=direction.constr
	H=direction.H
	k,k=H.shape
	include=np.array(k*[True])
	include[list(constr.fixed)]=False
	c_index, var_prop, includemap = decomposition(H, include)
	if c_index is None:
		return False
	constr.CI=c_index[-1]
	mc_problems=[]#list of [index,associate,condition index]
	for cix in range(1,len(c_index)):
		if c_index[-cix]>SMALL_COLLINEARITY:
			if np.sum(var_prop[-cix]>0.5)>1:
				var_prop_ix=np.argsort(var_prop[-cix])[::-1]
				var_prop_val=var_prop[-cix][var_prop_ix]
				j=var_prop_ix[0]
				j=includemap[j]
				done=var_prop_check(direction.panel,var_prop_ix, var_prop_val, includemap,j,mc_problems,c_index[-cix])
				if done:
					break
	return mc_problems

def var_prop_check(panel,var_prop_ix,var_prop_val,includemap,assc,mc_problems,cond_index):
	for i in range(1,len(var_prop_ix)):
		if var_prop_val[i]<0.5:
			return True
		index=var_prop_ix[i]
		index=includemap[index]
		mc_problems.append([index,assc,cond_index])
		return False
		
def add_mc_constraint(direction,mc_problems,mc_dict,newton_failed):
	"""Adds constraints for severe MC problems"""
	constr=direction.constr
	if newton_failed:
		limit=SMALL_COLLINEARITY
	else:
		limit=MAX_COLLINEARITY
	if len(mc_problems)==0:
		return False
	a=np.array(mc_problems)
	a=np.argsort(a[:,2])[::-1]
	for i in a:
		index,assc,cond_index=mc_problems[i]
		if (not index in mc_dict) and cond_index>SMALL_COLLINEARITY:
			mc_dict[index]=[assc,cond_index]#contains also collinear variables that are only slightly collinear, which shall be restricted when calcuating CV-matrix.	
		if not ((index in constr.associates) or (index in constr.collinears)) and cond_index>limit:
			constr.add(index,assc,'collinear')
			return True
	return False
		
		
def remove_all_multicoll(direction,ll,newton_failed):
	H=direction.H
	k,k=H.shape
	mc_dict=dict()
	for i in range(k):
		mc_problems=multicoll_problems(direction)
		remvd=add_mc_constraint(direction,mc_problems,mc_dict,newton_failed)
		if not remvd:
			break
	select_arma(direction.constr, ll)
	return mc_dict



def select_arma(constr,ll):
	for i in constr.collinears:
		if constr.constraints[i].category in ['rho','lambda', 'gamma','psi']:
			assc=constr.collinears[i]
			if not assc in constr.fixed:
				if abs(ll.args_v[assc])<abs(ll.args_v[i]):
					constr.delete(i)
					constr.add(assc,i,'collinear')
	
	
			
		


