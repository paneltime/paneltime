#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradient and Hessian construction for the panel ARIMA-GARCH likelihood.

This refactor keeps the original public API (``gradient``, ``hessian`` and
``T``), but makes the gradient side less repetitive and documents the shapes
that the Hessian helper functions expect.  The Hessian formulae themselves are
kept explicit because the block names are mathematically informative and useful
when comparing against numerical derivatives.
"""

import numpy as np

from .. import functions as fu
from . import calculus_functions_new as cf
from ..processing import arguments

EXTREME_VALUE = 1e100


def _clip_extreme(x):
	"""Clip extreme values while preserving ``None``."""
	if x is None:
		return None
	return np.clip(x, -EXTREME_VALUE, EXTREME_VALUE)


def _sum_over_panel(x):
	"""Sum a derivative array over the panel dimensions ``N`` and ``T``."""
	return np.sum(x, axis=(0, 1))


class gradient:
	"""Compute and store first derivatives used by the likelihood Hessian.

	The ``get`` method returns the total score vector ``g`` and the per-observation
	gradient array ``G``.  It also stores named derivative blocks on ``self`` so
	the Hessian class can reuse them without recomputing first derivatives.
	"""

	def __init__(self, panel):
		self.panel = panel

	def arima_grad(self, k, x, ll, sign, pre):
		"""Derivative of an ARIMA lag block.

		Returns ``None`` when the block has zero length.  Otherwise the result has
		shape ``(N, T, k)`` and is multiplied by the observation-inclusion mask.
		"""
		if k == 0:
			return None

		N, T, _ = x.shape
		x = fu.dotroll(pre, k, sign, x, ll).reshape(N, T, k)
		return _clip_extreme(x) * self.panel.included[3]

	def garch_arima_grad(self, ll, dRE, varname=None):
		"""Variance-equation derivative induced by an ARIMA error derivative.

		Returns three objects:
		``dvar_sigma``: GARCH variance-recursion derivative.
		``dvRE_dx``: random-effect variance derivative.
		``d_input``: demeaned random-effect input derivative.
		"""
		panel = self.panel
		d_input = 0
		dvRE_dx = None

		if panel.N > 1 and panel.options.fixed_random_group_eff > 0 and dRE is not None:
			d_eRE_sq = 2 * ll.e_RE * dRE
			dmean_e2 = panel.mean(d_eRE_sq, (0, 1))
			d_input = (d_eRE_sq - dmean_e2) * panel.included[3]
			dvRE_dx = dmean_e2 * panel.included[3]

		dvar_sigma = None
		if panel.pqdkm[4] > 0 and dRE is not None:
			dvar_sigma = fu.arma_dot(ll.GAR_1MA, cf.prod((ll.h_e_val, dRE)), ll)
			dvar_sigma = dvar_sigma * panel.included[3]

		return dvar_sigma, dvRE_dx, d_input

	def _store_arima_derivatives(self, ll, incl):
		"""Create derivatives of the residual equation for beta/rho/lambda."""
		panel = self.panel
		p, q, _, _, _ = panel.pqdkm

		self.X_RE = (
			panel.XIV
			+ ll.re_obj_i.RE(panel.XIV, panel)
			+ ll.re_obj_t.RE(panel.XIV, panel)
		) * incl

		self.de_rho_RE = self.arima_grad(p, ll.u_RE, ll, -1, ll.AMA_1)
		self.de_lambda_RE = self.arima_grad(q, ll.e_RE, ll, -1, ll.AMA_1)
		self.de_beta_RE = -fu.arma_dot(ll.AMA_1AR, self.X_RE, ll) * incl

	def _store_sigma_derivatives(self, ll):
		"""Create GARCH/random-effect derivatives linked to beta/rho/lambda."""
		for name in ("rho", "lambda", "beta"):
			dvar_sigma, dvRE, d_input = self.garch_arima_grad(
				ll, getattr(self, f"de_{name}_RE"), name
			)
			setattr(self, f"dvar_sigma_{name}", dvar_sigma)
			setattr(self, f"dvRE_{name}", dvRE)
			setattr(self, f"d_{name}_input", d_input)

	def _store_garch_derivatives(self, ll, N, T):
		"""Create direct derivatives of the variance equation."""
		panel = self.panel
		_, _, _, k, m = panel.pqdkm

		dG = np.array(panel.W_a)
		dG[:, 0, 0] = 0
		self.dvar_omega = fu.arma_dot(ll.GAR_1, dG, ll)

		self.dvar_initvar = None
		if arguments.INITVAR in ll.args.args_d:
			self.dvar_initvar = np.tile(ll.GAR_1[0], (N, 1)).reshape(N, T, 1)

		self.dvar_mu = cf.prod((ll.dvarRE_mu, panel.included[3])) if panel.N > 1 else None
		self.dvar_gamma = None
		self.dvar_psi = None
		self.dvar_z = None

		if m > 0:
			self.dvar_gamma = self.arima_grad(k, ll.llfunc.var, ll, 1, ll.GAR_1)
			self.dvar_psi = self.arima_grad(m, ll.h_val, ll, 1, ll.GAR_1)
			if ll.h_z_val is not None:
				self.dvar_z = fu.arma_dot(ll.GAR_1MA, ll.h_z_val, ll)

	def _score_block(self, de_RE, dvar_sigma, dLL_e, dLL_var):
		"""Score contribution from one residual/variance parameter block."""
		return cf.add((cf.prod((dvar_sigma, dLL_var)), cf.prod((de_RE, dLL_e))), True)

	def get(self, ll, dLL_e=None, dLL_var=None):
		"""Return the score vector and per-observation gradient array."""
		if dLL_var is None or dLL_e is None:
			dLL_var, dLL_e = ll.llfunc.gradient()

		self.dLL_e = dLL_e
		self.dLL_var = dLL_var

		N, T, _ = self.panel.X.shape
		incl = self.panel.included[3]

		self._store_arima_derivatives(ll, incl)
		self._store_sigma_derivatives(ll)
		self._store_garch_derivatives(ll, N, T)

		dLL_beta = self._score_block(self.de_beta_RE, self.dvar_sigma_beta, dLL_e, dLL_var)
		dLL_rho = self._score_block(self.de_rho_RE, self.dvar_sigma_rho, dLL_e, dLL_var)
		dLL_lambda = self._score_block(self.de_lambda_RE, self.dvar_sigma_lambda, dLL_e, dLL_var)
		dLL_gamma = cf.prod((self.dvar_gamma, dLL_var))
		dLL_psi = cf.prod((self.dvar_psi, dLL_var))
		dLL_omega = cf.prod((self.dvar_omega, dLL_var))
		dLL_initvar = cf.prod((self.dvar_initvar, dLL_var))
		dLL_mu = cf.prod((self.dvar_mu, dLL_var))
		dLL_z = cf.prod((self.dvar_z, dLL_var))

		G = cf.concat_marray((
			dLL_beta, dLL_rho, dLL_lambda, dLL_gamma, dLL_psi,
			dLL_omega, dLL_initvar, dLL_mu, dLL_z,
		))
		g = _sum_over_panel(G)

		return g, G


class hessian:
	"""Construct the analytical Hessian from stored gradient components.

	The class keeps the original public API, but the surrounding module now
	separates gradient construction and Hessian construction more clearly.
	"""

	def __init__(self,panel,g):
		self.panel=panel
		self.its=0
		self.g=g



	def get(self,ll,d2LL_de2,d2LL_dln_de,d2LL_dln2):	
		H = self.hessian(ll,d2LL_de2,d2LL_dln_de,d2LL_dln2)
		return H


	def hessian(self,ll,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		panel=self.panel

		g=self.g
		p,q,d,k,m=panel.pqdkm
		incl=self.panel.included[3]

		GARM=(ll.GAR_1,m,1)

		GARK=(ll.GAR_1,k,1)

		d2var_gamma2			=   cf.prod((2,  
							  		cf.dd_func_lags(panel,ll,GARK, 	g.dvar_gamma,			g.dLL_var,  transpose=True)))
		d2var_gamma_psi			=	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_psi,				g.dLL_var)
		d2var_gamma_rho			=	cf.dd_func_lags(panel,ll,GARK,	g.dvar_sigma_rho,		g.dLL_var)
		d2var_gamma_lambda		=	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_sigma_lambda,	g.dLL_var)
		d2var_gamma_beta		=	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_sigma_beta,	g.dLL_var)
		d2var_gamma_initvar 	= 	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_initvar,			g.dLL_var)
		d2var_gamma_omega 		= 	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_omega,			g.dLL_var)
		d2var_gamma_z			=	cf.dd_func_lags(panel,ll,GARK, 	g.dvar_z,				g.dLL_var)

		
		d2var_psi_rho					=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_rho_RE)),		g.dLL_var)
		d2var_psi_lambda			=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_lambda_RE)),	g.dLL_var)
		d2var_psi_beta				=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_beta_RE)),	g.dLL_var)
		d2var_psi_z						=	cf.dd_func_lags(panel,ll,GARM, 	ll.h_z_val,								g.dLL_var)

		AMAq=(ll.AMA_1,q,-1)
		d2var_lambda2,		d2e_lambda2		=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'lambda', transpose=True)
		d2var_lambda_rho,	d2e_lambda_rho	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'rho' )
		d2var_lambda_beta,	d2e_lambda_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'beta')

		AMAp=(ll.AMA_1,p,-1)
		d2var_rho_beta,		d2e_rho_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAp,	'rho',		'beta', u_gradient=True)



		d2var_mu_rho,d2var_mu_lambda,d2var_mu_beta,d2var_mu_z,mu=None,None,None,None,None
		if panel.N>1:
			d2var_mu_rho			=	cf.sumNT(cf.prod((ll.ddvarRE_mu_vRE, 	g.dvRE_rho,  	 	g.dLL_var)))
			d2var_mu_lambda			=	cf.sumNT(cf.prod((ll.ddvarRE_mu_vRE, 	g.dvRE_lambda,  	g.dLL_var)))
			d2var_mu_beta			=	cf.sumNT(cf.prod((ll.ddvarRE_mu_vRE, 	g.dvRE_beta,  	 	g.dLL_var)))
			d2var_mu_z=None
			d2var_mu2=0

		d2var_z_beta, d2var_z_lambda, d2var_z_rho, d2var_z2 = None, None, None, None
		if not ll.h_z_val is None:
			d2var_z2				=	sum(sum(fu.arma_dot(ll.GAR_1MA,ll.h_2z_val, ll), 0),1) 
			d2var_z_rho				=	cf.dd_func_z(ll, g.de_rho_RE)
			d2var_z_lambda			=	cf.dd_func_z(ll, g.de_lambda_RE)
			d2var_z_beta			=	cf.dd_func_z(ll, g.de_beta_RE)

		d2var_rho2,	d2e_rho2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'rho',		'rho' )
		d2var_beta2,d2e_beta2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'beta',		'beta')



		(de_rho_RE,de_lambda_RE,de_beta_RE)=(g.de_rho_RE,g.de_lambda_RE,g.de_beta_RE)
		(dvar_sigma_rho,dvar_sigma_lambda,dvar_sigma_beta)=(g.dvar_sigma_rho,g.dvar_sigma_lambda,g.dvar_sigma_beta)
		(dvar_mu,dvar_z)=(g.dvar_mu, g.dvar_z)		

		d2var_beta_omega, d2var_beta_initvar, d2var_rho_omega, d2var_lambda_omega=None, None, None, None
		d2var_rho_initvar, d2var_lambda_initvar = None, None
		


		#Final:
		D2LL_beta2 					=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dvar_sigma_beta, 	dvar_sigma_beta,	d2e_beta2, 					d2var_beta2)
		D2LL_beta_rho		      	=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dvar_sigma_beta, 	dvar_sigma_rho,		T(d2e_rho_beta), 		T(d2var_rho_beta))
		D2LL_beta_lambda			=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dvar_sigma_beta, 	dvar_sigma_lambda,	T(d2e_lambda_beta), 	T(d2var_lambda_beta))
		D2LL_beta_gamma				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	g.dvar_gamma,		None, 					T(d2var_gamma_beta))
		D2LL_beta_psi				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	g.dvar_psi,			None, 					T(d2var_psi_beta))
		D2LL_beta_omega				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	g.dvar_omega,		None, 					d2var_beta_omega)
		D2LL_beta_initvar			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	g.dvar_initvar,		None, 					d2var_beta_initvar)
		D2LL_beta_mu				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	dvar_mu,			None, 					d2var_mu_beta)
		D2LL_beta_z					=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dvar_sigma_beta, 	dvar_z,				None, 					T(d2var_z_beta))

		D2LL_rho2					=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dvar_sigma_rho, 	dvar_sigma_rho,		d2e_rho2, 					d2var_rho2)
		D2LL_rho_lambda				=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dvar_sigma_rho, 	dvar_sigma_lambda,	T(d2e_lambda_rho), 		T(d2var_lambda_rho))
		D2LL_rho_gamma				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dvar_sigma_rho, 	g.dvar_gamma,		None, 					T(d2var_gamma_rho))	
		D2LL_rho_psi				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dvar_sigma_rho, 	g.dvar_psi,			None, 					T(d2var_psi_rho))
		D2LL_rho_omega				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dvar_sigma_rho, 	g.dvar_omega,		None, 					d2var_rho_omega)
		D2LL_rho_initvar			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 	 	None,			dvar_sigma_rho, 	g.dvar_initvar,		None, 					d2var_rho_initvar)
		D2LL_rho_mu					=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dvar_sigma_rho, 	dvar_mu,			None, 					T(d2var_mu_rho))
		D2LL_rho_z					=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dvar_sigma_rho, 	dvar_z,				None, 					T(d2var_z_rho))

		D2LL_lambda2				=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dvar_sigma_lambda, 	dvar_sigma_lambda,	T(d2e_lambda2), 		T(d2var_lambda2))
		D2LL_lambda_gamma			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	g.dvar_gamma,		None, 					T(d2var_gamma_lambda))
		D2LL_lambda_psi				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	g.dvar_psi,			None, 					T(d2var_psi_lambda))
		D2LL_lambda_omega			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	g.dvar_omega,		None, 					d2var_lambda_omega)
		D2LL_lambda_initvar			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	g.dvar_initvar,		None, 					d2var_lambda_initvar)
		D2LL_lambda_mu				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	dvar_mu,			None, 					T(d2var_mu_lambda))
		D2LL_lambda_z				=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dvar_sigma_lambda, 	dvar_z,				None, 					T(d2var_z_lambda))

		D2LL_gamma2					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_gamma, 		g.dvar_gamma,		None, 					T(d2var_gamma2))
		D2LL_gamma_psi				=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dvar_gamma, 		g.dvar_psi,			None, 					d2var_gamma_psi)
		D2LL_gamma_omega			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_gamma, 		g.dvar_omega,		None, 					d2var_gamma_omega)
		D2LL_gamma_initvar			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_gamma,		g.dvar_initvar,		None, 					d2var_gamma_initvar)
		D2LL_gamma_mu				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_gamma, 		dvar_mu,			None, 					None)
		D2LL_gamma_z				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_gamma, 		dvar_z,				None, 					d2var_gamma_z)

		D2LL_psi2					=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dvar_psi, 		g.dvar_psi,			None, 					None)
		D2LL_psi_omega				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_psi, 		g.dvar_omega,		None, 					None)
		D2LL_psi_initvar			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_psi, 	 	g.dvar_initvar,		None, 					None)
		D2LL_psi_mu					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_psi, 		dvar_mu,			None, 					None)
		D2LL_psi_z					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_psi, 		dvar_z,				None, 					d2var_psi_z)

		D2LL_omega2					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_omega, 		g.dvar_omega,		None, 					None)
		D2LL_omega_initvar			=	cf.dd_func(None,		None,	 	 	d2LL_dln2,	None, 	 	  	None,			g.dvar_omega, 		g.dvar_initvar,		None, 					None)
		D2LL_omega_mu				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_omega, 		g.dvar_mu,			None, 					None)
		D2LL_omega_z				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_omega, 		g.dvar_z,			None, 					None)

		D2LL_initvar2				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dvar_initvar,		g.dvar_initvar,		None, 					None)
		D2LL_initvar_mu				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dvar_mu, 			g.dvar_initvar,		None, 					None)
		D2LL_initvar_z				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dvar_z, 			g.dvar_initvar,		None, 					None)

		D2LL_mu2					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dvar_mu, 			dvar_mu,			None, 					None)
		D2LL_mu_z					=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dvar_mu, 			dvar_z,				None, 					d2var_mu_z)

		D2LL_z2						=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dvar_z, 			dvar_z,				None, 					d2var_z2)




		H= [[D2LL_beta2,					D2LL_beta_rho,				D2LL_beta_lambda,				D2LL_beta_gamma,			D2LL_beta_psi,			D2LL_beta_omega,				D2LL_beta_initvar, 		D2LL_beta_mu,		D2LL_beta_z		],
				[T(D2LL_beta_rho),		D2LL_rho2,						D2LL_rho_lambda,				D2LL_rho_gamma,				D2LL_rho_psi,				D2LL_rho_omega,					D2LL_rho_initvar, 		D2LL_rho_mu,		D2LL_rho_z			],
				[T(D2LL_beta_lambda),	T(D2LL_rho_lambda),		D2LL_lambda2,						D2LL_lambda_gamma,		D2LL_lambda_psi,		D2LL_lambda_omega,			D2LL_lambda_initvar, 	D2LL_lambda_mu,	D2LL_lambda_z		],
				[T(D2LL_beta_gamma),	T(D2LL_rho_gamma),		T(D2LL_lambda_gamma),		D2LL_gamma2,					D2LL_gamma_psi,			D2LL_gamma_omega, 			D2LL_gamma_initvar, 	D2LL_gamma_mu,	D2LL_gamma_z		],
				[T(D2LL_beta_psi),		T(D2LL_rho_psi),			T(D2LL_lambda_psi),			T(D2LL_gamma_psi),		D2LL_psi2,					D2LL_psi_omega, 				D2LL_psi_initvar, 		D2LL_psi_mu,		D2LL_psi_z			],
				[T(D2LL_beta_omega),	T(D2LL_rho_omega),		T(D2LL_lambda_omega),		T(D2LL_gamma_omega),	T(D2LL_psi_omega),	D2LL_omega2, 						D2LL_omega_initvar, 	D2LL_omega_mu,	D2LL_omega_z		], 
				[T(D2LL_beta_initvar),T(D2LL_rho_initvar),	T(D2LL_lambda_initvar),	T(D2LL_gamma_initvar),T(D2LL_psi_initvar),T(D2LL_omega_initvar), 	D2LL_initvar2, 				D2LL_initvar_mu,D2LL_initvar_z		], 
				[T(D2LL_beta_mu),			T(D2LL_rho_mu),				T(D2LL_lambda_mu),			T(D2LL_gamma_mu),			T(D2LL_psi_mu),			T(D2LL_omega_mu), 			D2LL_initvar_mu, 			D2LL_mu2,				D2LL_mu_z			],
				[T(D2LL_beta_z),			T(D2LL_rho_z),				T(D2LL_lambda_z),				T(D2LL_gamma_z),			T(D2LL_psi_z),			T(D2LL_omega_z), 				D2LL_initvar_z, 			D2LL_mu_z,			D2LL_z2				]]

		H=cf.concat_matrix(H)
		if H[-1,-1]==0:
			H[-1,-1]=1
		#for debugging:
		if False:
			from .. import debug
			Hn=debug.hess_debug(ll,panel,g,0.00000001)#debugging
			

		self.its+=1
		if np.any(np.isnan(H)):
			return None
		#print(H[0]/1e+11)
		return H





def T(x):
	if x is None:
		return None
	return x.T