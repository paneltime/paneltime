#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import linesearch
from . import direction
from .computation import Computation
from .. import likelihood as logl

import numpy as np

#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
LR_LIMIT = 5.991	# chi2(2) 5% critical value: each reduction fixes one rho and one lambda
LR_LIMIT_GAMMA = 3.841	# chi2(1) 5% critical value: gamma ridge fixes one lag

def dfpmax(args, panel):
	"""Initialize and run one constrained likelihood optimization locally."""
	args = np.array(args)
	comput = Computation(args, panel, panel.options.tolerance, TOLX, True)
	for position, constraint in comput.constr.fixed.items():
		args[position] = constraint.value
	ll = logl.LL(args, panel, constraints=comput.constr, print_err=True)
	if ll.LL is None:
		print("WARNING: Initial arguments failed, attempting default OLS-arguments ...")
		panel.args.set_init_args(panel, default=True)
		ll = logl.LL(panel.args.args_OLS, panel, constraints=comput.constr, print_err=True)
		if ll.LL is None:
			raise RuntimeError("OLS-arguments failed too, you should check the data")
		print("default OLS-arguments worked")

	s = _state(comput, ll, panel)
	_, s['dx_norm'], _ = direction.get(s['g'], s['x'], s['H'], comput.constr, s['f'], s['hessin'], simple=False)
	pg = direction.projected_gradient(s['g'], s['x'], comput.constr)
	g_norm = np.max(np.abs(pg)*np.maximum(np.abs(s['x']), 1.0))/max(abs(s['f']), 1.0)
	s['its'], s['incr'], s['step'] = 0, 0.0, 1.0
	if g_norm < comput.gtol and comput.is_concave(s['H']):
		s['conv'], s['msg'] = 2, "Convergence at starting parameters"
	else:
		s = _iterate(comput, panel, s)

	# A converged ARMA fit whose AR and MA polynomials plausibly share a root is
	# refitted without the highest rho and lambda, kept only if the LR test accepts it.
	removed = []
	while s['conv'] in (1, 2):
		pair = comput.constr.arma_cancellation(s['x'], s['H'])
		if pair is None:
			break
		saved, saved_constr = s, comput.constr
		comput.arma_reduced += pair
		comput.set_constr(s['x'], panel.options.ARMA_constraint)
		x = s['x'].copy()
		x[pair] = 0.0
		ll_try = logl.LL(x, panel, constraints=comput.constr, print_err=True)
		if ll_try is None or ll_try.LL is None or not np.isfinite(ll_try.LL):
			comput.arma_reduced = comput.arma_reduced[:-2]
			comput.constr = saved_constr
			s = dict(saved, its=s['its'])
			break
		r = _state(comput, ll_try, panel)
		r.update(its=s['its'], incr=0.0, step=1.0, dx_norm=s['dx_norm'])
		r = _iterate(comput, panel, r)
		if r['conv'] not in (1, 2) or 2*(saved['f'] - r['f']) > LR_LIMIT:
			comput.arma_reduced = comput.arma_reduced[:-2]
			comput.constr = saved_constr
			s = dict(saved, its=r['its'])
			break
		s = r
		removed += pair
	if removed:
		names = ", ".join(panel.args.names_v[i] for i in sorted(removed))
		s['msg'] += f". AR/MA common factor removed: {names} fixed at zero"

	# Singular gamma ridges are treated similarly: constrain the highest gamma
	# lag only when the Wald test flags a singular split and the LR test accepts it.
	# Stalled fits are included, since a flat ridge is a common cause of stalling.
	gamma_removed = []
	ridge_wald = panel.options.GARCH_ridge_wald
	while ridge_wald > 0 and s['conv'] in (1, 2, 4):
		lag = comput.constr.gamma_ridge(s['x'], s['H'], ridge_wald)
		if lag is None:
			break
		saved, saved_constr = s, comput.constr
		comput.gamma_reduced += lag
		comput.set_constr(s['x'], panel.options.ARMA_constraint)
		x = s['x'].copy()
		x[lag] = 0.0
		ll_try = logl.LL(x, panel, constraints=comput.constr, print_err=True)
		if ll_try is None or ll_try.LL is None or not np.isfinite(ll_try.LL):
			comput.gamma_reduced = comput.gamma_reduced[:-1]
			comput.constr = saved_constr
			s = dict(saved, its=s['its'])
			break
		r = _state(comput, ll_try, panel)
		r.update(its=s['its'], incr=0.0, step=1.0, dx_norm=s['dx_norm'])
		r = _iterate(comput, panel, r)
		if r['conv'] not in (1, 2) or 2*(saved['f'] - r['f']) > LR_LIMIT_GAMMA:
			comput.gamma_reduced = comput.gamma_reduced[:-1]
			comput.constr = saved_constr
			s = dict(saved, its=r['its'])
			break
		s = r
		s, comput, released = _release_gamma(comput, panel, s, saved)
		if released:
			names = ", ".join(panel.args.names_v[i] for i in lag)
			s['msg'] += f". GARCH ridge restriction released: {names}"
			break
		gamma_removed += lag
	if gamma_removed:
		names = ", ".join(panel.args.names_v[i] for i in sorted(gamma_removed))
		s['msg'] += f". GARCH ridge regularized: {names} fixed at zero"

	# Only retain numerical results needed for inference, not loop/server state.
	return {
		'x': s['x'], 'f': s['f'], 'll': s['ll'], 'g': s['g'], 'G': s['G'], 'H': s['H'],
		'constr': comput.constr, 'conv': s['conv'], 'its': s['its'], 'msg': s['msg'],
		'incr': s['incr'], 'dx_norm': s['dx_norm'],
	}


def _release_gamma(comput, panel, reduced, original):
	"""Retry the latest gamma reduction once, within the remaining iteration budget."""
	if reduced['its'] >= panel.options.max_iterations:
		return reduced, comput, False
	retry = Computation(reduced['x'], panel, panel.options.tolerance, TOLX, True)
	retry.arma_reduced = comput.arma_reduced.copy()
	retry.gamma_reduced = comput.gamma_reduced[:-1]
	retry.set_constr(reduced['x'], panel.options.ARMA_constraint)
	ll = logl.LL(reduced['x'].copy(), panel, constraints=retry.constr, print_err=True)
	if ll is None or ll.LL is None or not np.isfinite(ll.LL):
		return reduced, comput, False
	r = _state(retry, ll, panel)
	r.update(its=reduced['its'], incr=0.0, step=1.0, dx_norm=reduced['dx_norm'])
	r = _iterate(retry, panel, r)
	keep = (
		r['conv'] in (1, 2)
		and np.isfinite(r['f'])
		and r['f'] >= max(original['f'], reduced['f'])
		and np.all(np.isfinite(r['x']))
		and np.all(np.isfinite(r['g']))
		and retry.is_concave(r['H'])
		and retry.constr.gamma_ridge(r['x'], r['H'], panel.options.GARCH_ridge_wald) is None
	)
	if keep:
		return r, retry, True
	return dict(reduced, its=r['its']), comput, False


def _state(comput, ll, panel):
	g, G = comput.calc_gradient(ll)
	H = comput.calc_hessian(ll)
	try:
		hessin = np.linalg.inv(H)
	except np.linalg.LinAlgError:
		hessin = -np.identity(len(g)) * panel.args.init_var
	return {'x': ll.args.args_v, 'f': ll.LL, 'll': ll, 'g': g, 'G': G, 'H': H, 'hessin': hessin}


def _iterate(comput, panel, s):
	"""Runs quasi-Newton iterations from state s until a convergence code is set."""
	armaconstr = panel.options.ARMA_constraint
	MAXITER = panel.options.max_iterations
	x, f, g, G, H, hessin, ll = s['x'], s['f'], s['g'], s['G'], s['H'], s['hessin'], s['ll']
	step, its, incr = s['step'], s['its'], s['incr']
	conv, msg = 0, f"No convergence within {MAXITER} iterations"
	for iteration in range(its, MAXITER):
		dx, _, _ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel, ll, step)
		ls.lnsrch(x, f, g, H, dx)
		step = ls.step
		dx_realized = ls.x - x
		incr = ls.f - f
		ll = ls.ll
		x, f, hessin, H, G, g, conv, _ = comput.exec(
			dx_realized, hessin, H, incr, iteration, ls, armaconstr
		)
		its = iteration + 1

		if conv==1:
			msg = "Convergence on negligible expected gain"
		elif conv==2:
			msg = "Convergence on negligible projected gradient"
		elif conv==3:
			msg = f"No convergence within {MAXITER} iterations"
		elif conv==4:
			msg = "Maximization has stalled"
		elif conv==5:
			msg = "Collinear"
		elif conv==6:
			msg = "Overflow"
		elif its == MAXITER:
			conv = 3
			msg = f"No convergence within {its} iterations"

		if conv > 0:
			break
	if conv == 0:
		conv = 3
	return dict(s, x=x, f=f, g=g, G=G, H=H, hessin=hessin, ll=ll, step=step,
				its=its, incr=incr, conv=conv, msg=msg)