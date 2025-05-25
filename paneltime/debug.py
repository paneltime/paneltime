#!/usr/bin/env python
# -*- coding: utf-8 -*-

#used for debugging


from . import functions as fu
from .likelihood import main as lgl
import numpy as np
import time
import os



def hess_debug(ll,panel,g,d):
	"""Calculate the Hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debugged gradient"""
	x=ll.args.args_v
	n=len(x)
	dx=np.identity(n)*d
	H=np.zeros((n,n))
	ll0=lgl.LL(x,panel)
	f0=g.get(ll0)
	for i in range(n):
		ll=lgl.LL(x+dx[i],panel)
		if not ll is None:
			f1=g.get(ll)
			H[i]=(f1-f0)/d


	return H

def grad_debug(ll,panel,d):
	"""Calcualtes the gradient numerically. For debugging"""
	x=ll.args.args_v
	n=len(x)
	dx=np.abs(x.reshape(n,1))*d
	dx=dx+(dx==0)*d
	dx=np.identity(n)*dx

	g=np.zeros(n)
	f0=lgl.LL(x,panel)
	for i in range(n):
		for j in range(5):
			dxi=dx[i]*(0.5**j)
			f1=lgl.LL(x+dxi,panel)
			if not f1 is None:
				if not f1.LL is None:
					g[i]=(f1.LL-f0.LL)/dxi[i]
					break
	return g



def grad_debug_detail(f0,panel,d,llname,varname1,pos1=0):
	args1=lgl.copy_array_dict(f0.args.args_d)
	args1[varname1][pos1]+=d

	f0=lgl.LL(f0.args.args_d, panel)
	f1=lgl.LL(args1, panel)

	if type(llname)==list or type(llname)==tuple:
		ddL=(f1.__dict__[llname[0]].__dict__[llname[1]]-f0.__dict__[llname[0]].__dict__[llname[1]])/d
	else:
		ddL=(f1.__dict__[llname]-f0.__dict__[llname])/d
	return ddL

def test_c_armas(u_RE, var, e_RE, panel, ll, G):
	var2 = fu.arma_dot(ll.GAR_1, G,ll) + fu.arma_dot(ll.GAR_1MA, ll.llfunc.h_val,ll)
	var2 = var2*panel.included[3]
	e_RE2 = fu.arma_dot(ll.AMA_1AR,u_RE,ll)	
	print(f"Testsums arma: c:{np.sum(var**2)}, py:{np.sum(var2**2)}")
	print(f"Testsums e: c:{np.sum(e_RE**2)}, py:{np.sum(e_RE2**2)}")



def hess_debug_detail(f0,panel,d,llname,varname1,varname2,pos1=0,pos2=0):
	args1=lgl.copy_array_dict(f0.args.args_d)
	args2=lgl.copy_array_dict(f0.args.args_d)
	args3=lgl.copy_array_dict(f0.args.args_d)
	args1[varname1][pos1]+=d
	args2[varname2][pos2]+=d	
	args3[varname1][pos1]+=d
	args3[varname2][pos2]+=d
	f1=lgl.LL(args1, panel)
	f2=lgl.LL(args2, panel)
	f3=lgl.LL(args3, panel)
	if type(llname)==list:
		ddL=(f3.__dict__[llname[0]].__dict__[llname[1]]-f2.__dict__[llname[0]].__dict__[llname[1]]
										 -f1.__dict__[llname[0]].__dict__[llname[1]]+f0.__dict__[llname[0]].__dict__[llname[1]])/(d**2)
	else:
		ddL=(f3.__dict__[llname]-f2.__dict__[llname]-f1.__dict__[llname]+f0.__dict__[llname])/(d**2)
	return ddL



def LL_calc(self,panel):
	raise RuntimeError('The function needs ot be updated to match the new LL_calc function')
	return


def save_reg_data(ll, panel, fname = 'repr.csv', heading=True):
	#saves data neccessary to reproduce 
	N,T,k = panel.X.shape
	N,T,m = panel.W_a.shape
	heading = [f'X{i}' for i in range(k)]
	heading += [f'W{i}' for i in range(m)]
	heading += ['Y', 'e','u', 'var', 'LL','AMA_1AR', 'GAR_1MA' , 'GAR_1', 'coefs_names', 'coefs']
	a = np.concatenate((panel.X, panel.W_a, panel.Y, ll.e, ll.u, ll.llfunc.var), 2)
	a = a.reshape((T, a.shape[2]))
	coefs = np.zeros((T,1))
	coef_names = np.array([['']]*T, dtype='<U20')
	coef_names[:]=''
	coefs[:len(ll.args.args_v),0] = ll.args.args_v
	coef_names[:len(ll.args.args_v),0] = ll.args.names_v
	a = np.concatenate((a, 
														ll.llfunc.ll_value[0].reshape((T,1)),
																								ll.AMA_1AR[0].reshape((T,1))[::-1],
																								ll.GAR_1MA[0].reshape((T,1))[::-1],
																								ll.GAR_1[0].reshape((T,1))[::-1],
																								coef_names,
																								coefs
																								),1)
	a = np.concatenate(([heading], a),0)
	np.savetxt(fname, a, fmt='%s', delimiter=';')	



def test_date_map(arrayize, timevar, N, T, sel, panel):
	tvar = arrayize(timevar, N, panel.max_T, T, panel.idincl,sel)
	a=[tvar[i][0][0] for i in panel.date_map]
	if not np.all(a==np.sort(a)):	
		raise RuntimeError("It seems the data is not properly sorted on time")
		
