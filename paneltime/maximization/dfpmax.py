#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import linesearch
from . import direction
import sys

import numpy as np
import time

#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5

def dfpmax(x, f, g, hessin, H, comput, panel, slave_id, ll, armaconstr, slave_server):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

	MAXITER = panel.options.max_iterations
	
	
	fdict = {}
	step = 1.0

	dx, dx_norm, H_ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)

	for its in range(MAXITER):  	#Main loop over the iterations.

		res = calc(g, x, H, comput, f, hessin, 
							panel, step, its, fdict, ll, armaconstr, dx)
			

		(g, G, x, H, comput, f, ll, hessin, 
			conv, incr, step, g_norm, dx) = res

		
		terminate = srvr_terminated(slave_server, its)
		
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
		elif terminate or its + 1 == MAXITER:
			msg = f"No convergence within {its} iterations" 

		if terminate or (conv>0):
			break

	constr = comput.constr
	v = vars()
	ret = {k:v[k] for k in v if not k in ['panel', 'comput', 'ls']}
	return ret

def srvr_terminated(slave_server, its):
	if slave_server is None:
		return False
	print('server up, receiving request')
	kill = slave_server.kill_request()
	print(f'server up, requested kill={kill}')
	return kill



def calc(g, x, H, comput, f, hessin, panel, step, its, fdict, ll, armaconstr, dx):
		dx, dx_norm, H_ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel, ll, step)
		ls.lnsrch(x, f, g, H, dx)	

		step = ls.step
		dx_realized = ls.x - x
		incr = ls.f - f
		fdict[its] = ls.f
		ll = ls.ll

		x, f, hessin, H, G, g, conv, g_norm, dx = comput.exec(dx_realized,  hessin, H, incr, its, ls, armaconstr)


		#print(list(ls.x[1:3])+list(ls.x[-2:])+[comput.ci, len(comput.constr.mc_constr)])
		#print(g)


		return g, G, x, H, comput, f, ll, hessin, conv, incr, step, g_norm, dx