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
	c_u = stat.correlogram(panel, u, 2, center=True)[1:]
	c_u2 = stat.correlogram(panel, u*u*panel.included[3], 2, center=True)[1:]
	rho0,lmbda0= ARMA_process_calc(c_u, p, q)
	psi0, gamma0 = GARCH_process_calc(c_u2 - c_u**2, k, m)
	v = panel.var(u) 

	vreg = panel.h_func(0, v, v)

	initvar = vreg
	if panel.options.EGARCH:
		omega = vreg*0.2
	else:
		# Unconditional variance v_e = omega/(1-psi-gamma), v_e being the innovation variance implied by the ARMA start values
		omega = vreg*innovation_var_factor(rho0, lmbda0, p, q)*(1 - psi0 - gamma0)
		initvar = vreg*innovation_var_factor(rho0, lmbda0, p, q)
		# The returned v is the fixed first-period variance, so it must be that of the innovations, not of the ARMA process
		v = v*innovation_var_factor(rho0, lmbda0, p, q)

	return beta,rho0,lmbda0, psi0, gamma0, v, initvar, omega




def ARMA_process_calc(c, p, q):
	"""Moment estimates of the first AR and MA coefficients from the autocorrelations c=(r1, r2)"""
	r1, r2 = c
	if p > 0 and q > 0:
		# ARMA(1,1): r2 = rho*r1 and r1 = (1+rho*l)(rho+l)/(1+2*rho*l+l**2); the latter is solved
		# for the invertible root of (r1-rho)*l**2 + (2*r1*rho-1-rho**2)*l + (r1-rho) = 0
		if abs(r1) < 1e-3:
			return 0.0, 0.0
		rho = np.clip(r2/r1, -0.9, 0.9)
		a = r1 - rho
		b = 2*r1*rho - 1 - rho**2
		disc = max(b*b - 4*a*a, 0.0)
		lmda = 2*a/(-b + np.sqrt(disc))
		return float(rho), float(np.clip(lmda, -0.9, 0.9))
	if p > 0:
		return float(np.clip(r1, -0.9, 0.9)), 0.0
	if q > 0:
		r1 = np.clip(r1, -0.49, 0.49)
		# r1 = l/(1+l**2), invertible root
		lmda = 0.0 if abs(r1) < 1e-8 else (1 - np.sqrt(1 - 4*r1**2))/(2*r1)
		return 0.0, float(lmda)
	return 0.0, 0.0


def GARCH_process_calc(c, k, m):
	"""Moment estimates of the first ARCH (psi) and GARCH (gamma) coefficients from the
	autocorrelations c=(r1, r2) of the squared residuals, net of the autocorrelation implied by the mean equation.
	For GARCH(1,1) with pi=psi+gamma: r2 = pi*r1 and r1 = psi*(1-pi**2+pi*psi)/(1-pi**2+psi**2)."""
	default = (0.1, 0.5)
	r1, r2 = c
	if m == 0:
		return default
	if k == 0:
		return float(np.clip(r1, 0.01, 0.9)), 0.0
	if r1 < 0.02 or r2 <= 0:
		return default
	pi = np.clip(r2/r1, 0.3, 0.97)
	s = 1 - pi**2
	a = r1 - pi
	if a >= 0:
		return default
	disc = max(s*s - 4*a*r1*s, 0.0)
	psi = np.clip((s - np.sqrt(disc))/(2*a), 0.01, min(0.5, pi - 0.05))
	return float(psi), float(pi - psi)


def innovation_var_factor(rho, lmda, p, q):
	"""Ratio of the innovation variance to the variance of an ARMA(1,1) with the given coefficients"""
	if p > 0 and q > 0:
		return (1 - rho**2)/(1 + 2*rho*lmda + lmda**2)
	if p > 0:
		return 1 - rho**2
	if q > 0:
		return 1/(1 + lmda**2)
	return 1.0






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
