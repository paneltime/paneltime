#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import functions as fu


def add_static_constraints(constr,panel,ll,its):
	
	add_custom_constraints(constr,panel.user_constraints,ll)
	general_constraints=[('rho',-2,2),('lambda',-2,2),('gamma',-2,2),('psi',-2,2)]
	add_custom_constraints(constr,general_constraints,ll)
	if panel.loadargs==False:
		if panel.m_zero:
			constr.add(panel.args.positions['psi'][0],None,'psi=1 constr',value=1)
			if panel.k>0 and panel.k>0 and (panel.p>0 or panel.q>0) and its<3:
				constr.add(panel.args.positions['gamma'][0],None,'gamma=0 constr',value=0)
		else:
			if panel.p>0 or panel.q>0:
				if its<3 and panel.m>0:
					constr.add(panel.args.positions['psi'][0],None,'psi=0 constr',value=0)				
				if panel.k>0 and its<5:
					constr.add(panel.args.positions['gamma'][0],None,'psi=0 constr',value=0)
			elif panel.k>0 and its<3:
				constr.add(panel.args.positions['gamma'][0],None,'gamma=0 constr',value=0)
		

	
	
def add_dynamic_constraints(G,panel,ll,constr,dx_norm,hessian,its,old_constr):

	remove_all_multicoll(panel,dx_norm,hessian, constr,old_constr)
	remove_singularities(constr, hessian)


	
def remove_singularities(constr,hessian):
	n=len(hessian)
	habs=np.abs(hessian)
	m=np.max(habs)
	h_temp=np.array(hessian)
	for i in range(n):
		if i not in constr.fixed:
			if np.linalg.det(h_temp)>0:
				return
			zeros=len(np.nonzero(habs[i]/m<1e-100)[0])
			if zeros>=n-1:
				constr.add(i,None,'singularity')
				h_temp[i,i]=-m
		

def add_custom_constraints(constr,constraints,ll):
	"""Adds custom range constraints\n\n
		constraints shall be on the format [(name, minimum, maximum), ...]"""		
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
	
	
def remove_one_multicoll(panel,H,constr):
	limit=1000
	k,k=H.shape
	include=np.array(k*[True])
	include[list(constr.fixed)]=False
	c_index, var_prop, includemap = decomposition(H, include)
	if c_index is None:
		return False
	constr.CI=c_index[-1]	
	if c_index[-1]>10000:
		constr.jump_start_variable=np.nonzero(var_prop[-1]==np.max(var_prop[-1]))[0][0]
	else:
		constr.jump_start_variable=None
	for cix in range(1,len(c_index)):
		if c_index[-cix]<limit:
			return False
		
		if np.sum(var_prop[-cix]>0.5)>1:
			var_prop_ix=np.argsort(var_prop[-cix])[::-1]
			var_prop_val=var_prop[-cix][var_prop_ix]
			j=var_prop_ix[0]
			j=includemap[j]
			for i in range(1,len(var_prop_ix)):
				if var_prop_val[i]<0.5:
					return False
				assc=var_prop_ix[i]
				assc=includemap[assc]
				if (j in constr.associates) or (j in constr.collinears):
					return False
				else:
					constr.add(j,assc,'collinear')
					return True
	return False

def remove_all_multicoll(panel,dx_norm,H,constr,old_constr):
	k,k=H.shape
	for i in range(k):
		remvd=remove_one_multicoll(panel,H,constr)
		if not remvd:
			break
	select_multicoll_with_biggest_direction(constr,dx_norm,old_constr)


def select_multicoll_with_biggest_direction(constr,dx_norm,old_constr):
	if dx_norm is None:
		return	
	max_dx0=[0,None,None]
	if len(constr.collinears)==0:
		return
	for i in constr.collinears:
		dx_abs0=abs(dx_norm[i])
		if dx_abs0>max_dx0[0]:
			assc0=constr.collinears[i]
			max_dx0=[dx_abs0,i,assc0]
	
	
	
	mc=np.array(list(constr.collinears.keys()))
	dx_abs=np.abs(dx_norm[mc])
	max_dx=mc[np.argsort(dx_abs)]
	try:
		list(old_constr.collinears.keys())[0]
	except:
		assc=constr.collinears[max_dx[0]]
		constr.delete(max_dx[0])
		constr.add(assc,max_dx[0],'collinear')
		return
	for i in max_dx:
		old_mc=np.sort(list(old_constr.collinears.keys()))
		curr_mc=fu.copy_array_dict(constr.collinears)
		assc=constr.constraints[i].assco_ix
		curr_mc[assc]=i
		if assc in curr_mc:
			curr_mc.pop(i)
		curr_mc=np.sort(list(curr_mc.keys()))
		swithch=len(old_mc)!=len(curr_mc)
		if not swithch:
			swithch=not np.all(old_mc==curr_mc)
		if swithch:
			constr.delete(i)
			constr.add(assc,i,'collinear')
			return



			
	
	
			
		


