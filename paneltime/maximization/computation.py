#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-iteration computations for the ML optimizer: gradient, Hessian,
constraints, search direction and convergence tests.

Conventions (shared with direction.py and linesearch.py): the log-likelihood
LL is MAXIMIZED, g is its gradient and H its Hessian (negative definite at a
proper maximum). H is never modified here; non-concavity is handled when the
search direction is computed, so H can still be used for standard errors.

Convergence codes (conv)
  0  continue
  1  expected gain of the constrained Newton step is negligible (H concave)
  2  projected gradient is negligible
  3  maximum number of iterations
  4  stalled: repeated negligible steps and no function increase
  5  divergence: non-finite or exploding gradient/residuals
"""

from ..output import stat_dist
from . import constraints
from .. import likelihood as logl
from . import direction
import sys
import numpy as np


EPS = 3.0e-16
TOLX = 4*EPS
STPMX = 100.0


class Computation:
	def __init__(self, args, panel, gtol, tolx, grestricted):
		self.gradient = logl.calculus.gradient(panel)
		self.gtol = panel.options.tolerance
		self.tolx = tolx
		self.hessian = logl.hessian(panel, self.gradient)
		self.panel = panel
		self.ci = 0
		self.mc_report = {}
		self.H_correl_problem = False
		self.singularity_problems = False
		self.H, self.g, self.G = None, None, None
		self.mcollcheck = False
		self.rec = []
		self.quit = False
		self.avg_incr = 0
		self.errs = []
		self.CI_anal = 2
		self.pqdkm = panel.pqdkm
		self.init_arma_its = 0
		self.grestricted = grestricted
		self.set_constr(args, panel.options.ARMA_constraint)
		

	# ------------------------------------------------------------ constraints

	def set(self, its, increment, lmbda, rev, H, ll, x, armaconstr):
		self.its = its
		self.lmbda = lmbda
		self.has_reversed_directions = rev
		self.increment = increment
		self.constr_old = self.constr
		self.constr = constraints.Constraints(self.panel, x, its, armaconstr)
		self.constr.add_static_constraints(self, its, ll)
		self.constr.multicoll_report(H, self.panel.options.multicoll_threshold_report)
		self.ci = self.constr.ci
		self.mc_report = self.constr.mc_report
		self.singularity_problems = len(self.mc_report) > 0

	def set_constr(self, args, armaconstr):
		self.constr_old = None
		self.constr = constraints.Constraints(self.panel, args, 0, armaconstr)
		self.constr.add_static_constraints(self, 0)

	def fixed_constr_change(self):
		return set(self.constr.fixed.keys()) != set(self.constr_old.fixed.keys())

	# -------------------------------------------------------------- iteration

	def exec(self, dx_realized, hessin, H, incr, its, ls, armaconstr):
		f, x, g_old, ll = ls.f, ls.x, ls.g, ls.ll

		g, G = self.calc_gradient(ll)
		hessin, H = self.hessin_get(g, g_old, dx_realized, ll, hessin, H, its)
		self.set(its, incr, ls.alam, ls.rev, H, ll, x, armaconstr)

		# Constrained, concavity-safe Newton direction at the new point.
		# Variables at an active bound get (practically) zero step, so no
		# masking with the previous line search's constraints is needed.
		dx, dx_norm, _ = direction.get(g, x, H, self.constr, f, hessin, simple=False)

		# Convergence measures, all relative to |LL|:
		scale = max(abs(f), 1.0)
		pg = direction.projected_gradient(g, x, self.constr)		# KKT gradient
		g_norm = min(np.max(np.abs(pg)*np.maximum(np.abs(x), 1.0))/scale, 1e+50)
		pgain, totpgain = potential_gain(dx, g, H)					# model gain of the step
		gain_rel = totpgain/scale
		concave = self.is_concave(H)
		
		step_rel = np.max(np.abs(dx_realized)/np.maximum(np.abs(x), 1.0))
		self.errs.append(step_rel < 10000*TOLX)

		if not self.panel.options.supress_output:
			print(f"its:{its}, f:{f}, gnorm:{g_norm:.3e}, |g|:{np.max(np.abs(g)):.3e}, "
				  f"|pg|:{np.max(np.abs(pg)):.3e}, gain:{totpgain:.3e}, "
				  f"max_pgain:{np.max(pgain):.3e}, concave:{concave}, alam:{ls.alam}, "
				  f"ls:{ls.conv}")
			sys.stdout.flush()

		min_its = 0#sum(self.pqdkm[:2]) + sum(self.pqdkm[3:]) + 6
		if self.diverged(g, ll):
			conv = 5
		elif its < min_its:
			conv = 0
		elif concave and gain_rel < self.gtol:
			conv = 1
		elif g_norm < self.gtol:
			conv = 2
		elif its >= self.panel.options.max_iterations:
			conv = 3
		elif sum(self.errs[-3:]) == 3 and incr < 1e-15:
			conv = 4
		else:
			conv = 0

		return x, f, hessin, H, G, g, conv, g_norm, dx

	def is_concave(self, H):
		"""True if H is negative definite on the non-fixed parameters."""
		free = np.ones(len(H), dtype=bool)
		if self.constr is not None:
			free[list(self.constr.fixed.keys())] = False
		Hf = np.asarray(H, dtype=float)[free][:, free]
		if Hf.size == 0:
			return True
		if not np.all(np.isfinite(Hf)):
			return False
		return np.linalg.eigvalsh(0.5*(Hf + Hf.T)).max() < 0

	@staticmethod
	def diverged(g, ll):
		e = np.asarray(ll.e)
		return (not np.all(np.isfinite(g)) or not np.all(np.isfinite(e))
				or np.max(np.abs(g)) > 1e+50 or np.max(np.abs(e)) > 1e+50)

	# ------------------------------------------------------ gradient, Hessian

	def calc_gradient(self, ll):
		dLL_lnv, DLL_e = ll.llfunc.gradient()
		self.LL_gradient_tobit(ll, DLL_e, dLL_lnv)
		g, G = self.gradient.get(ll, DLL_e, dLL_lnv)
		return g, G

	def calc_hessian(self, ll):
		d2LL_de2, d2LL_dln_de, d2LL_dln2 = ll.llfunc.hessian()
		self.LL_hessian_tobit(ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)
		return self.hessian.get(ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)

	def LL_gradient_tobit(self, ll, DLL_e, dLL_lnv):
		sgn = [1, -1]
		self.f = [None, None]
		self.f_F = [None, None]
		for i in [0, 1]:
			if self.panel.tobit_active[i]:
				I = self.panel.tobit_I[i]
				self.f[i] = stat_dist.norm(sgn[i]*ll.e_norm[I], cdf=False)
				self.f_F[i] = (ll.F[i] != 0)*self.f[i]/(ll.F[i] + (ll.F[i] == 0))
				DLL_e[I] = sgn[i]*self.f_F[i]*ll.llfunc.v_inv05[I]
				dLL_lnv[I] = -0.5*DLL_e[I]*ll.e_RE[I]

	def LL_hessian_tobit(self, ll, d2LL_de2, d2LL_dln_de, d2LL_dln2):
		sgn = [1, -1]
		if sum(self.panel.tobit_active) == 0:
			return
		lf = ll.llfunc
		e1s1 = ll.e_norm
		e2s2 = ll.e2*lf.v_inv
		e3s3 = e2s2*e1s1
		e1s2 = e1s1*lf.v_inv05
		e1s3 = e1s1*lf.v_inv
		e2s3 = e2s2*lf.v_inv05
		f_F = self.f_F
		for i in [0, 1]:
			if self.panel.tobit_active[i]:
				I = self.panel.tobit_I[i]
				f_F2 = f_F[i]**2
				d2LL_de2[I] = -(sgn[i]*f_F[i]*e1s3[I] + f_F2*lf.v_inv[I])
				d2LL_dln_de[I] = 0.5*(f_F2*e1s2[I] + sgn[i]*f_F[i]*(e2s3[I] - lf.v_inv05[I]))
				d2LL_dln2[I] = 0.25*(f_F2*e2s2[I] + sgn[i]*f_F[i]*(e1s1[I] - e3s3[I]))

	def hessin_get(self, g, g_old, dx_realized, ll, hessin_orig, H_orig, its):
		"""Analytical Hessian and its inverse. H is returned unmodified;
		non-concavity is handled in direction.py."""
		H = self.calc_hessian(ll)
		if H is None or not np.all(np.isfinite(H)):
			# calculus returns None for a NaN Hessian: keep the previous one
			H = H_orig if H_orig is not None else -np.eye(len(g))
		hessin = hess_inv(H, hessin_orig)
		return hessin, H

# ================================================================ functions

def det_managed(H):
	try:
		return np.linalg.det(H)
	except Exception:
		return 1e+100


def inv_hess(hessian):
	try:
		return -np.linalg.inv(hessian)
	except Exception:
		return None


def condition_index(H):
	n = len(H)
	d = np.maximum(np.abs(np.diag(H)).reshape((n, 1)), 1e-30)**0.5
	C = -H/(d*d.T)
	ev = np.abs(np.linalg.eigvalsh(0.5*(C + C.T)))**0.5
	if min(ev) == 0:
		return 1e-150
	return max(ev)/min(ev)


def hess_inv(h, hessin):
	"""Inverse of h; pseudo-inverse if singular; the old hessin if both fail."""
	try:
		return np.linalg.inv(h)
	except Exception:
		pass
	try:
		return np.linalg.pinv(h)
	except Exception as e:
		print(e)
		return hessin


def potential_gain(dx, g, H):
	"""Quadratic-model gain of the full step dx, and for each variable the
	loss of gain from leaving that variable out (all others included).
	Vectorized closed form of the original loop:
	  gain_i = g_i dx_i + dx_i (H dx)_i - 0.5 H_ii dx_i^2"""
	dx = np.asarray(dx, dtype=float).ravel()
	g = np.asarray(g, dtype=float).ravel()
	H = np.asarray(H, dtype=float)
	Hdx = H @ dx
	full = g @ dx + 0.5*(dx @ Hdx)
	gain_i = g*dx + dx*Hdx - 0.5*np.diag(H)*dx**2
	return np.minimum(np.abs(gain_i), 1e+50), min(abs(full), 1e+50)