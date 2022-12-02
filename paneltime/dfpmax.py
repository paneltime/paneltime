import numpy as np
import time
import direction
import linesearch
#import stat_functions as stat


#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5

def dfpmax(x, comput, callback, panel, maxiter):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

	x, ll, f, g, hessin, H = comput.calc_init_dir(x)

	its, msg = 0, ''
	MAXITER = 10000
	
	cbhandler = CallBackHandler(callback, maxiter)
	

	for its in range(MAXITER):  	#Main loop over the iterations.
		constr = comput.constr

		dx, dx_norm = direction.get(g, x, H, constr, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel)
		ls.lnsrch(x, f, g, dx)	

		dx = ls.x - x
		incr = ls.f - f


		x, f, hessin, H, g, conv = comput.exec(dx, dx_norm,  hessin, H, ls.f, ls.x, ls.g, incr, ls.rev, ls.alam, its, ls.ll)

		err = np.max(np.abs(dx)) < TOLX
		
		terminated = conv or err or its+1==MAXITER
		print(f"terminated: {terminated}, quit: {callback.outbox['quit']}")
		if conv:
			msg = "Convergence on zero gradient; local or global minimum identified"
		elif err:
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
		elif terminated:
			msg = "No convergence within %s iterations" %(MAXITER,)
			
		
		cbhandler.assign(ls, msg, dx_norm, f, x, H, comput.G, g, hessin, dx, 
						  incr, its, comput.constr, 'linesearch', terminated, 
						  conv)			

		if terminated:		
			return
			
class CallBackHandler:
	def __init__(self, callback, maxiter):
		self.t = time.time()
		self.f_maxiter = None
		self.callback = callback
		self.maxiter = maxiter
															
	def assign(self, ls, msg, dx_norm, f, x, H, G, g, hessin, dx, incr, its, 
						  constr, task, terminated, conv):
		
		if self.callback.outbox['quit']:
			raise RuntimeError(callback.QUIT_EXCEPTION)
	
		if its==self.maxiter:
			self.f_maxiter = f
			
		if its>self.maxiter:
			a=0

		if (time.time()-self.t < 0.1) and (not terminated) and (its!=self.maxiter):
			return
		
		if msg == '':
			msg = ls.msg	

		self.callback.callback(msg = msg, dx_norm = dx_norm, f = f, x = x, 
				 H = H, G=G, g = g, hessin = hessin, dx = dx, 
				 incr = incr, rev = ls.rev, alam = ls.alam, 
				 its = its, constr = constr, perc=min(its/100, 1), task = task, 
				 f_maxiter = self.f_maxiter, terminated=terminated, conv = conv)
		
		return

