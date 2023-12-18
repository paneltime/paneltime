#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import linesearch
from . import direction

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

def dfpmax(x, f, g, hessin, H, comput, panel, slave_id, ll):
  """Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
  Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
  its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
  gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
  iter (the number of iterations that were performed), and fret (the minimum value of the
  function). The routine lnsrch is called to perform approximate line minimizations.
  fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """


  its, msg = 0, ''
  MAXITER = 10000
  
  

  fdict = {}
  for its in range(MAXITER):  	#Main loop over the iterations.


    dx, dx_norm, H_ = direction.get(g, x, H, comput.constr, f, hessin, simple=False)
    ls = linesearch.LineSearch(x, comput, panel, ll)
    ls.lnsrch(x, f, g, H, dx)	

    dx_realized = ls.x - x
    incr = ls.f - f
    fdict[its] = ls.f
    ll = ls.ll


    x, f, hessin, H, G, g, conv, se, det, anal = comput.exec(dx_realized,  hessin, H, incr, its, ls)

    err = (np.max(np.abs(dx_realized)) < TOLX) and (its >len(comput.constr.constr_matrix)+2)

    terminate = (conv>0) or err or its+1==MAXITER

    print(f"sid:{slave_id}, f:{ls.f}, conv:{conv}, its:{its}")

    if conv==1:
      msg = "Convergence on zero gradient; local or global minimum identified"
    elif conv==2:
      msg = "Convergence on zero expected gain; local or global minimum identified given multicolinearity constraints"		
    elif conv==3:
      msg = "Reached the maximum number of iterations"		  
    elif err:
      msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
    elif terminate:
      msg = "No convergence within %s iterations" %(MAXITER,)
    
    if terminate or (conv>0) or err:
      break
    
  constr = comput.constr
  v = vars()
  ret = {k:v[k] for k in v if not k in ['panel', 'comput', 'ls']}
  return ret


