#!/usr/bin/env python
# -*- coding: utf-8 -*-

#calculates the arma matrices for GARCH and ARIMA

from ctypes import create_unicode_buffer
from pathlib import Path
import numpy as np
import os
from .. import cfunctions
import sys
from . import hfunc

def set_garch_arch(panel,args,u, G):
	"""Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
	the diagonal band"""
	N, T, _ = u.shape
	rho = np.insert(-args['rho'].flatten(),0,1)
	psi = args['psi']
	psi = np.insert(args['psi'].flatten(),0,0)
	_,_,lags,_,_ = panel.options.pqdkm

	lmbda = args['lambda'].flatten()
	gmma =  -args['gamma'].flatten()

	z = 0
	if panel.z_active:
		z = args['z']

	parameters = np.array(( N , T , 
									len(lmbda), len(rho), len(gmma), len(psi), 
									panel.options.EGARCH, z))

	AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h = inv_c(parameters, lmbda, rho, gmma, psi, N, T, u, G, panel.T_arr, 
												 panel.h_func_cpp)
	r=[]
	#Creating nympy arrays with name properties. 
	for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
		x = np.maximum(np.minimum(locals()[i], 1e+150), -1e+150)
		m = round(x, panel)
		r.append((m,i))
	for i in ['e', 'var', 'h']:
		r.append(locals()[i])
	return r



def inv_c(parameters,lmbda, rho,gmma, psi, N, T, u, G, T_arr, h_expr):

	AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h=(
	np.append([1],np.zeros(T-1)),
				np.zeros(T),
				np.append([1],np.zeros(T-1)),
				np.zeros(T),
				np.zeros((N,T,1)),
				np.zeros((N,T,1)),
				np.zeros((N,T,1))
		)
 
	u = np.array(u,dtype = np.float64)
	
	T_arr = np.array(T_arr.flatten(),dtype = float)
	cfunctions.armas(parameters, lmbda, rho, gmma, psi, 
										AMA_1, AMA_1AR, GAR_1, GAR_1MA, 
										u, e, var, h, G, T_arr, h_expr)   

	return AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h



def round(arr, panel):
	#skiping round as it makes the code hang for some reson (see below)
	#There may be small differences in calculation between different systems. 
	#For consistency, the inverted matrixes are slightly rounded
	n_digits = panel.options.ARMA_round
	arr2 = arr*(np.abs(arr)>1e-100)
	digits = 14-np.floor(np.log10(np.abs(arr2 + (arr2==0))))
	scale = 10**digits
	return np.round(arr2*scale)/scale



	