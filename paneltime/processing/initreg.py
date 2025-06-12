#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains the argument class for the panel object
from ..output import stat_functions as stat
from .. import likelihood as logl

from .. import random_effects as re
from .. import functions as fu

import numpy as np



def start_values(panel, X = None, Y = None):
	
	p, q, d, k, m=panel.pqdkm

	if X is None:
		X = panel.X
		Y = panel.Y
	gfre=panel.options.fixed_random_group_eff
	tfre=panel.options.fixed_random_time_eff
	re_obj_i=re.re_obj(panel,True,panel.T_i,panel.T_i,gfre)
	re_obj_t=re.re_obj(panel,False,panel.date_count_mtrx,panel.date_count,tfre)

	X=(X+re_obj_i.RE(X, panel)+re_obj_t.RE(X, panel))*panel.included[3]
	Y=(Y+re_obj_i.RE(Y, panel)+re_obj_t.RE(Y, panel))*panel.included[3]
	beta,u=stat.OLS(panel,X,Y,return_e=True)
	rho,lmbda=ARMA_process_calc(u,panel)
	psi, gamma = 0.05, 0.95
	v = panel.var(u) 
	v = v
	vreg = panel.h_func(0, v, v)


	if panel.options.include_initvar:
		initvar = vreg*0.25
		omega = 0
	else:
		initvar = vreg
		omega = vreg

	return beta,rho,lmbda, psi, gamma,v, initvar, omega




def ARMA_process_calc(e,panel):
	return 0,0
	c=stat.correlogram(panel,e,2,center=True)[1:]
	if abs(c[0])<0.1:
		return 0,0
	
	rho = 0.5*(c[0] + c[1]/c[0])
	if abs(rho)>0.99:
		return 0,0

	lmbda = 0
	den = 2*(c[0]-rho)
	rtexp = ( (rho**2 - 1)*(rho**2 - 1 + 4*c[0]**2 - 4*c[0]*rho) )
	if den!=0 and rtexp>0:
		lmbda1 = (1 - 2*c[0]*rho + rho**2)/den
		lmbda2 = (rtexp**0.5) / den

		if abs(lmbda1+lmbda2)>abs(lmbda1-lmbda2):
			lmbda = max(min(lmbda1 - lmbda2,0.99), -0.99)
		else:
			lmbda = max(min(lmbda1 + lmbda2,0.99), -0.99)


	rho = max(min(rho,0.5), -0.5)
	lmbda = max(min(lmbda,0.5), -0.5)
	return rho,lmbda






def set_GARCH(panel,initargs,u,m):
	matrices=logl.set_garch_arch(panel,initargs)
	if matrices is None:
		e=u
	else:
		AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
		e = fu.dot(AMA_1AR,u)*panel.included[3]		
	h=h_func(e, panel,initargs)
	if m>0:
		initargs['gamma'][0]=0
		initargs['psi'][0]=0


def h_func(e,panel,initargs):
	z=None
	if len(initargs['z'])>0:
		z=initargs['z'][0][0]
	h_val,h_e_val,h_2e_val,h_z,h_2z,h_e_z=logl.h(e,z,panel)
	return h_val*panel.included[3]
