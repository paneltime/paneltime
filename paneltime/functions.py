#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import ctypes as ct
import os
from . import cfunctions
import warnings

def save_csv(fname, array, sep = ','):
	f = open(fname, 'wt')
	try:
		for line in array:
			f.write(sep.join(line))
	except TypeError as e:
		np.savetxt(fname,array,fmt='%s', delimiter=sep)
	f.close()

import numpy as np

def solve(H, g, cond_limit=1e12, lam0=None, max_iter=20):
    """
    Solve H x =  g. If H is ill-conditioned, add damping: H + lambda*I.
    """

    if not np.isfinite(H).all() or not np.isfinite(g).all():
        raise ValueError("H or g contains NaN/Inf")

    n = H.shape[0]
    I = np.eye(n)

    cond = np.linalg.cond(H)

    # Try ordinary Newton step first
    if cond < cond_limit:
        try:
            return np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            pass

    # Damped solve
    if lam0 is None:
        lam = 1e-8 * np.linalg.norm(H, ord=2)
    else:
        lam = lam0

    for _ in range(max_iter):
        try:
            H_damped = H + lam * I
            return np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            lam *= 10

    # Fallback
    return np.linalg.lstsq(H, g, rcond=None)[0]


def dot(a,b,reduce_dims=True):
	"""Matrix multiplication. Returns the dot product of a*b where either a or be or both to be
	arrays of matrices. Faster than mmult, less general and only used for special purpose.
	Todo: generalize and merge"""
	if len(a.shape)==3 and len(b.shape)==2:  
		k,m = b.shape 
		N,T,k = a.shape    
		#x = np.dot(a.reshape((N*T,k)),b).reshape((N,T,m))
		x = np.array([np.dot(a[i],b) for i in range(a.shape[0])])
	elif len(a.shape)==3 and len(b.shape)==3:
		N,T,m = b.shape 
		N,T,k = a.shape
		#x = np.dot(a.reshape((N*T,k)).T,b.reshape((N*T,m)))
		x = np.sum([np.dot(a[i].T,b[i]) for i in range(a.shape[0])],0)   
	elif len(a.shape)==2 and len(b.shape)==2:
		if a.shape[1] == b.shape[0]:
			x = np.dot(a,b)
	elif len(a.shape)==2 and len(b.shape)==3:
		#this should only be called for differencing in panel.py. Must be checke if used by other processes
		N,T,k = b.shape 
		x = np.array([np.dot(a,b[:,:,i].T) for i in range(k)])
		x = x.swapaxes(0,2)
	return x

	

def dotroll(aband,k,sign,b,ll):
	x = sign*fast_dot_c(aband, b)
	w=[]
	for i in range(k):
		w.append(np.roll(np.array(x),i+1,1))
		w[i][:,:i+1]=0
	x=np.array(w)
	x=np.moveaxis(x,0,2)
	return x



def arma_dot(a,b,ll):
	if len(a)>2:#then this is a proper matrix
		(aband,k,sgn)=a
		if k==0:
			return None
		return dotroll(aband, k, sgn, b, ll)
	x = fast_dot_c(a, b)
	return x
	

import time
def fast_dot_c(a,b):
	a, name = a
	r = a[0]*b
	s0 =b.shape
	b = b.swapaxes(1,len(s0)-1)
	r = r.swapaxes(1,len(s0)-1)
	s1 = b.shape
	
	r = r.reshape(-1)
	b = b.reshape(-1)

	cols = int(np.prod(s1[:-1]))

	r = cfunctions.fast_dot(r, a, b, cols)

	r = r.reshape(s1)
	r = r.swapaxes(1,len(s0)-1)
	
	return r
							

	
def try_warn(function, args):
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		res = function(*args)
	return res