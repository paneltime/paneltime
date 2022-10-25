#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def get(g, x, H, constr, hessin, simple=True):
	n = len(x)
	if simple or (H is None):
		dx = -(np.dot(hessin,g.reshape(n,1))).flatten()
	else:
		#self.dx=solve(self.H, self.g, args.args_v, self.constr)
		dx = solve_delete(constr, H, g, x)	
	dx_norm = normalize(dx, x)
	
	return dx, dx_norm
	


def solve(H, g, x, constr):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		raise RuntimeError('Hessian is None')
	dx_init=-np.linalg.solve(H,g).flatten()	
	if constr is None:
		return dx_init	
	n=len(H)
	k=len(constr)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,(k)*[0])
	
	for i in range(k):
		H[n+i,n+i]=1
	j=0
	dx=np.zeros(len(g))
	for i in constr.fixed:
		#kuhn_tucker(constr.fixed[i],i,j,n, H, g, x,dx, recalc=False)
		kuhn_tucker2(constr.fixed[i],i,j,n, H, g, x,dx, dx_init, recalc=False)
		j+=1
	dx=-np.linalg.solve(H,g).flatten()	
	OK=False
	w=0
	for r in range(50):
		j2=j
		
		for i in constr.intervals:
			dx=kuhn_tucker(constr.intervals[i],i,j2,n, H, g, x,dx)
			j2+=1
		OK=constr.within(x+dx[:n],False)
		if OK: 
			break
		if r==k+3:
			#print('Unable to set constraints in computation calculation')
			break

	return dx[:n]


def solve_delete(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		raise RuntimeError('Cant solve with no coefficient matrix')
	try:
		list(constr.keys())[0]
	except:
		return -np.linalg.solve(H,g).flatten()	
	
	m=len(H)
	
	idx=np.ones(m,dtype=bool)
	delmap=np.arange(m)
	if len(list(constr.fixed.keys()))>0:#removing fixed constraints from the matrix
		idx[list(constr.fixed.keys())]=False
		H=H[idx][:,idx]
		g=g[idx]
		delmap-=np.cumsum(idx==False)
		delmap[idx==False]=m#if for some odd reason, the deleted variables are referenced later, an out-of-bounds error is thrown
	n=len(H)
	k=len(constr.intervals)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,(k)*[0])
	
	for i in range(k):
		H[n+i,n+i]=1
	dx=-np.linalg.solve(H,g).flatten()	
	xi_full=np.zeros(m)
	OK=False
	keys=list(constr.intervals.keys())
	for r in range(50):		
		for j in range(k):
			key=keys[j]
			dx=kuhn_tucker_del(constr,key,j,n, H, g, x,dx,delmap)
		xi_full[idx]=dx[:n]
		OK=constr.within(x+xi_full,False)
		if OK: 
			break
		if r==k+3:
			#print('Unable to set constraints in computation calculation')
			break
	xi_full=np.zeros(m)
	xi_full[idx]=dx[:n]
	return xi_full

def kuhn_tucker_del(constr,key,j,n,H,g,x,dx,delmap,recalc=True):
	q=None
	c=constr.intervals[key]
	i=delmap[key]
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx


def kuhn_tucker(c,i,j,n,H,g,x,dx,recalc=True):
	q=None
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx


def kuhn_tucker2(c,i,j,n,H,g,x,dx,dx_init,recalc=True):
	if c.assco_ix is None:
		return kuhn_tucker(c,i,j,n,H,g,x,dx,recalc)
	q=None
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx



	
def normalize(dx, x):
	dx_norm=(x!=0)*dx/(np.abs(x)+(np.abs(x)<1e-100))
	dx_norm=(x<1e-2)*dx+(x>=1e-2)*dx_norm	
	return dx_norm	