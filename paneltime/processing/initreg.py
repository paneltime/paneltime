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
	re_obj_i=re.REObj(panel,True,panel.T_i,panel.T_i,gfre)
	re_obj_t=re.REObj(panel,False,panel.date_count_mtrx,panel.date_count,tfre)

	X=(X+re_obj_i.RE(X, panel)+re_obj_t.RE(X, panel))*panel.included[3]
	Y=(Y+re_obj_i.RE(Y, panel)+re_obj_t.RE(Y, panel))*panel.included[3]
	beta,u=stat.OLS(panel,X,Y,return_e=True)
	rho0,lmbda0= ARMA_process_calc(u,panel)
	psi0, gamma0 = 0.1,0.5
	v = panel.var(u) 

	vreg = panel.h_func(0, v, v)

	initvar = vreg*0.7
	omega = vreg*0.2

	return beta,rho0,lmbda0, psi0, gamma0, v, initvar, omega




def ARMA_process_calc(e,panel):

	c=stat.correlogram(panel,e,2,center=True)[1:]
	if c[1]<0:
		rho, lmda = 0, c[0]
	else:
		rho = c[1]**0.5
		lmda = c[0]-rho
	return rho, lmda






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
