#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..output import stat_functions as stat
import numpy as np
import itertools



def get(g, x, H, constr, f, hessin, simple=True):
	n = len(x)
	if simple or (H is None):
		dx = -(np.dot(hessin,g.reshape(n,1))).flatten()
	else:
		dx, H, applied_constraints = solve(constr, H, g, x, f)	
	dx_norm = normalize(dx, x)
	if np.any(np.isnan(dx_norm)):
		if simple or (H is None):
			dx = -(np.dot(hessin,g.reshape(n,1))).flatten()
		else:
			dx, H, applied_constraints = solve(constr, H, g, x, f)	
		dx_norm = normalize(dx, x)
	return dx, dx_norm, H

def new(g, x, H, constr, f,dx, alam):
	n = len(x)
	dx, slope, rev = slope_check(g, dx)
	if constr is None:
		return dx*alam, slope, rev, []
	elif len(constr.intervals)==0:
		return dx*alam, slope, rev, []
	elif constr.within(x + dx):
		return dx*alam, slope, rev, []

	dxalam, H, applied_constraints = solve(constr, H, g*alam, x, f)	
	if np.sum(g*dxalam)<0.0:
		dxalam, H, applied_constraints = solve(constr, H, -g*alam, x, f)	
		slope = np.sum(g*dx)
		rev = True

	return dxalam, slope, rev, applied_constraints

def slope_check(g, dx):
	rev = False
	slope=np.sum(g*dx)					#Scale if attempted step is too big.
	if slope <= 0.0:
		dx=-dx
		slope=np.sum(g*dx)
		rev = True  
	return dx, slope, rev

def solve(constr,H, g, x, f):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""

	if H is None:
		raise RuntimeError('Cant solve with no coefficient matrix')
	try:
		list(constr.keys())[0]
	except:
		dx = -np.linalg.solve(H, g)
		return dx, H, []

	H_orig = np.array(H)
	m=len(H)

	n, g, H, delmap, keys, dx, idx = add_constraints(constr, H, g)
	k=len(keys)

	xi_full=np.zeros(m)
	OK=False
	
	applied_constraints=[]
	for j in range(k):
		xi_full[idx]=dx[:n]
		OK=constr.within(x+xi_full,False)
		if OK:break 
		key=keys[j]
		try:
			dx, H = kuhn_tucker(constr,key,j,n, H, g, x, f,dx,delmap, OK)
		except np.linalg.LinAlgError as e:
			if not 'Singular matrix' in e.args:
				raise np.linalg.LinAlgError(e)
			dx[delmap[key]] = 0
		applied_constraints.append(key)
	xi_full=np.zeros(m)
	xi_full[idx]=dx[:n]
	H_full = np.zeros((m,m))
	idx = idx.reshape((1,m))
	nz = np.nonzero(idx*idx.T)
	H_full[nz] = H[np.nonzero(np.ones((n,n)))]
	H_full[H_full==0] = H_orig[H_full==0]
	return xi_full, H_full, applied_constraints

def add_constraints(constr, H, g):
	#testing the direction in case H is singular (should be handled by constr, but 
	#sometimes it isn't), and removing randomly until H is not singular

	include=np.ones(len(g),dtype=bool) #all are initially included
	include[list(constr.fixed.keys())]=False #but not fixed constraints
	include_others = np.ones(sum(include), dtype=bool) #potential for adding additional ad hoc constraints by setting other variables to False
	include_copy = np.array(include) 
	include_copy[include] = include_others

	n, g_new, H_new, delmap, keys =  remove_and_enlarge(constr, H, g, include_copy)

	dx = -np.linalg.solve(H_new, g_new)

	for k in list(keys):
		if not include_copy[k]:#Remove all keys (interval constraints) that are all ready removed by 'False' in idx (fixed constraints)
			keys.pop(keys.index(k))
	return n, g_new, H_new, delmap, keys, dx, include_copy
	
	
def remove_and_enlarge(constr, H, g, idx):
	("Variables with fixed constraints are *removed*"
  	 "Slack variables are added for variables with interval constraints, which *enlarges* the hessian and gradient")
	#idx are fixed constraints
	#keys is constructed from interval constraints
	m = len(g)
	delmap=np.arange(m) 
	if not np.all(idx==True):#removing fixed constraints from the matrix
		
		H=H[idx][:,idx]
		g=g[idx]
		delmap-=np.cumsum(idx==False) #delmap is the list of indicies to the undeleted variables, after deletion
		delmap[idx==False]=m#if for some odd reason, the deleted variables are referenced later, an out-of-bounds error will be thrown  
		
	n=len(H)
	keys=list(constr.intervals.keys())
	k=len(keys)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,np.zeros(k))
	
	for i in range(k):
		H[n+i,n+i]=1

	return n, g, H, delmap, keys
	
def kuhn_tucker(constr,key,j,n,H,g,x, f, dx,delmap, OK,recalc=True):
	H = np.array(H)
	q=None
	c=constr.intervals[key]
	i=delmap[key]
	if not c.value is None:
		q=-(c.value-x[key])
	elif x[key]+dx[i]<c.min:
		q=-(c.min-x[key])
	elif x[key]+dx[i]>c.max:
		q=-(c.max-x[key])
	if q!=None:
		if OK:
			a=0
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx = -np.linalg.solve(H, g)
	return dx, H

def normalize(dx, x):
	x = np.abs(x)
	dx_norm=(x!=0)*dx/(x+(x<1e-100))
	dx_norm=(x<1e-2)*dx+(x>=1e-2)*dx_norm	
	return dx_norm	



