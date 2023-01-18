import numpy as np
import time
import direction
import linesearch
import callback
#import stat_functions as stat


#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5

def dfpmax(x, comput, callback, panel, slave_id):
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
	
	cbhandler = CallBackHandler(callback, slave_id)
	fdict = {}
	for its in range(MAXITER):  	#Main loop over the iterations.
		
		
		constr = comput.constr
		dx, dx_norm = direction.get(g, x, H, constr, f, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel)
		ls.lnsrch(x, f, g, dx)	

		dx = ls.x - x
		incr = ls.f - f
		fdict[its] = ls.f
		
		
		x, f, hessin, H, g, conv = comput.exec(dx, dx_norm,  hessin, H, ls.f, ls.x, ls.g, incr, ls.rev, ls.alam, its, ls.ll)
		
		err = np.max(np.abs(dx)) < TOLX
		
		terminate = (conv>0) or err or its+1==MAXITER
		
		#print(f"sid:{slave_id}, f:{ls.f}, conv:{conv}, its:{its}")

		if conv==1:
			msg = "Convergence on zero gradient; local or global minimum identified"
		elif conv==2:
			msg = "Convergence on zero expected gain; local or global minimum identified given multicolinearity constraints"		
		elif err:
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
		elif terminate:
			msg = "No convergence within %s iterations" %(MAXITER,)
			

		cbhandler.assign(ls, msg, dx_norm, f, x, H, comput.G, g, hessin, dx, 
						  incr, its, comput.constr, 'linesearch', terminate, 
						  conv, fdict)			

		if terminate or cbhandler.quit:	
			print(f"quit slave {slave_id}")
			return cbhandler.callback.outbox
			
class CallBackHandler:
	def __init__(self, callback, slave_id):
		self.t = time.time()
		self.callback = callback
		self.id = slave_id
		self.quit = False
		
															
	def assign(self, ls, msg, dx_norm, f, x, H, G, g, hessin, dx, incr, its, 
						  constr, task, terminate, conv, fdict):


		self.check_for_quit_order()
		


		if msg == '':
			msg = ls.msg	
		
		self.callback.callback(msg = msg, dx_norm = dx_norm, f = f, x = x, 
				 H = H, G=G, g = g, hessin = hessin, dx = dx, 
				 incr = incr, rev = ls.rev, alam = ls.alam, 
				 its = its, constr = constr, perc=min(its/100, 1), task = task, 
				 terminate= terminate or self.quit, conv = conv, slave_id = self.id, 
				 fdict = dict(fdict) )
		
	def check_for_quit_order(self):

		if not 'quit' in  self.callback.inbox:
			return
		if not self.callback.inbox['quit']:
			return
		self.quit = True		
		
	
				


