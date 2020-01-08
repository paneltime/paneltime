#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat


def add_initial_constraints(constr,panel,user_constraints,ll,its):
	
	add_custom_constraints(constr,user_constraints,ll)
	general_constraints=[('rho',-2,2),('lambda',-2,2),('gamma',-2,2),('psi',-2,2)]
	add_custom_constraints(constr,general_constraints,ll)
	if panel.m_zero and panel.loadargs==False:
		for eq in range(panel.args.n_equations):
			if its<1:
				constr.add(panel.args.positions['rho'][eq][0],None,'initial rho constr',value=0.3)
			if its<3:
				constr.add(panel.args.positions['gamma'][eq][0],None,'initial gamma constr',value=0.1)
			constr.add(panel.args.positions['psi'][eq][0],None,'psi=0 constr',value=0.1)

	
	
def add_constraints(G,panel,ll,constr,dx_conv,dx_conv_old,hessian,its,user_constraints):

	#remove_constants(panel, G,constr,ll)
	#remove_correl(panel, G, constr)
	remove_all_multicoll(panel,hessian, constr)
	remove_singularities(constr, hessian)
	a=0
	#remove_H_correl(panel,hessian,constr,True)
	
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
	if ll is None and  (type(minimum)==str or type(maximum)==str):
		return
	if ll is None:
		constr.add_named(name,None,'user constraint', [minimum,maximum])
		return
	for l in ll.lls:
		m=[minimum,maximum]
		for i in range(2):
			if type(m[i])==str:
				m[i]=eval(m[i],globals(),l.__dict__)
				[minimum,maximum]=m
				constr.add_named(name,None,'user constraint', [minimum,maximum],l.id)				




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
					value=0.5*(c.min+c.max)
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
				

		return True
	
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
		return True
		

	def add_all_named(self,name,assco,cause,interval,equation=None):
		try:
			positions=self.panel.args.positions[name]
		except KeyError as e:
			try:
				i=self.panel.args.name_positions_map[name]
			except  KeyError as e:
				print("Unable to set constraints on %s, the name does not exist" %(name,))
				return
			self.add(i,assco,cause, interval)
		if not equation is None:
			positions=[positions[equation]]
		for rng in positions:
			for i in range(len(rng)):
				self.add(i,assco,cause, interval)
				
				
	def add_named(self,name,assco,cause,interval,equation=None):
		try:
			positions=self.panel.args.positions[name]
		except KeyError as e:
			try:
				p=self.panel.args.name_positions_map[name]
			except  KeyError as e:
				print("Unable to set constraints on %s, the name does not exist" %(name,))
				return
			positions=[range(p,p+1)]
		for rng in positions:
			for i in range(len(rng)):
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
		


def remove_constants(panel,G,constr,ll):
	N,T,k=G.shape
	v=panel.var(G,(0,1))
	for i in range(1,k):
		if v[i]==0:
			constr.add(i,None,'constant')

def remove_H_correl(panel,hessian,constr,replace):
	k,k=hessian.shape
	include=np.array(k*[True])
	include[list(constr.fixed)]=False	
	if panel.has_intercept:
		include[0]=False
		
	hessian_abs=np.abs(hessian)
	x=(np.diag(hessian_abs)**0.5).reshape((1,k))
	x=(x.T*x)
	corr=hessian_abs/(x+(x==0)*1e-100)	
	for i in range(k):
		m=np.max(corr[i])
		if m>2*corr[i,i]:
			j=np.nonzero(corr[i]==m)[0][0]
			corr[:,j]=0
			corr[j,:]=0
			corr[j,j]=1	
	for i in range(k):
		corr[i,i:]=0

			

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>=1.0)[0]]

	IDs=correl_IDs(p)
	acc=None
	for i in IDs:
		for j in range(len(i)):
			if not i[j] in constr.fixed:
				acc=i.pop(j)
				break
		if not acc is None:
			for j in i:
				constr.add(j,acc,'h-correl',replace=replace)


def remove_correl(panel,G,constr):
	threshold=0.99
	N,T,k=G.shape
	include=np.array(k*[True])
	if False:
		include[list(constr.fixed)]=False
		if panel.has_intercept:
			include[0]=False
	corr=np.abs(stat.correl(G,panel))
	for i in range(k):
		corr[i,i:]=0
		if not include[i]:
			corr[:,i]=0

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>threshold)[0]]
	principal_factors=[]
	IDs=correl_IDs(p)
	for i in IDs:
		for j in range(len(i)):
			if not i[j] in constr.fixed:
				acc=i.pop(j)
				break
		for j in i:
			constr.add(j,acc,'correl')


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
	
	
def remove_one_multicoll(panel,H,constr):
	limit=100
	k,k=H.shape
	include=np.array(k*[True])
	include[list(constr.fixed)]=False
	C,includemap=normalize(H, include)
	c_index,var_prop=stat.var_decomposition(XXNorm=C)
	try:
		c_index,var_prop=stat.var_decomposition(XXNorm=C)
	except:
		return False
	zeros=np.zeros(len(c_index))
	c_index=c_index.flatten()
	constr.CI=c_index[-1]
	if c_index[-1]>limit:
		if np.sum(var_prop[-1]>0.5)>1:
			j=np.argsort(var_prop[-1])[-1]
			j=includemap[j]
			assc=np.argsort(var_prop[-1])[-2]
			assc=includemap[assc]
			constr.add(j,assc,'collinear')
			return True
	return False

def remove_all_multicoll(panel,H,constr):
	k,k=H.shape
	for i in range(k):
		remvd=remove_one_multicoll(panel,H,constr)
		if not remvd:
			return


