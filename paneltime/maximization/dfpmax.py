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

	x, f = ll.args.args_v, ll.LL
	g, G = comput.calc_gradient(ll)
	H = comput.calc_hessian(ll)
	try:
		hessin = np.linalg.inv(H)
	except np.linalg.LinAlgError:
		hessin = -np.identity(len(g)) * panel.args.init_var

	armaconstr = panel.options.ARMA_constraint
	MAXITER = panel.options.max_iterations
	step = 1.0
	_, dx_norm, _ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)

	for its in range(MAXITER):
		dx, _, _ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel, ll, step)
		ls.lnsrch(x, f, g, H, dx)
		step = ls.step
		dx_realized = ls.x - x
		incr = ls.f - f
		ll = ls.ll
		x, f, hessin, H, G, g, conv, _, _ = comput.exec(
			dx_realized, hessin, H, incr, its, ls, armaconstr
		)

		if conv==1:
			msg = "Convergence on zero gradient; local or global minimum identified"
		elif conv==2:
			msg = "Convergence on zero expected gain; local or global minimum identified given multicolinearity constraints"		
		elif conv==3:
			msg = "Reached the maximum number of iterations. This should not happen."		  
		elif conv==4:
			msg = "Maximization has stalled"
		elif conv==5:
			msg = "Collinear"
		elif conv==6:
			msg = "Overflow"
		elif its + 1 == MAXITER:
			msg = f"No convergence within {its} iterations" 

		if conv > 0 and its > 2:
			break

	# Only retain numerical results needed for inference, not loop/server state.
	return {
		'x': x, 'f': f, 'll': ll, 'g': g, 'G': G, 'H': H,
		'constr': comput.constr, 'conv': conv, 'its': its, 'msg': msg,
		'incr': incr, 'dx_norm': dx_norm,
	}