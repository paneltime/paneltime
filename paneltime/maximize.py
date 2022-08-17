import numpy as np
import time
import loglikelihood as logl
import computation
import direction
import linesearch


#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-2

def dfpmin(x, comput, callback, mp, panel):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """


	x, ll, f, g, hessin, H = comput.calc_init_dir(x)

	its = 0
	max_iter = 1000
	for its in range(max_iter):  	#Main loop over the iterations.
		constr = comput.constr
		dx, dx_norm = direction.get(g, x, H, constr, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, mp, panel)
		ls.lnsrch(x, f, g, dx)
		
		dx = ls.x - x
		incr = ls.f - f


		x, f, hessin, H, g, conv = comput.exec(dx, dx_norm,  hessin, H, ls.f, ls.x, ls.g, incr, ls.rev, ls.alam, its, ls.ll)
		
		
		callback(msg = ls.msg, dx_norm = dx_norm, f = f, x = x, H = H, g = g, 
				hessin = hessin, dx = dx, incr = incr, rev = ls.rev, alam = ls.alam, 
				its = its, constr = comput.constr, perc = 1.0, task = 'Line search')			
		

		if conv:  
			msg = "Convergence on zero gradient; local or global minimum identified"
			return f,x,H,its,1, ls, msg, dx_norm, incr #FREEALL
		
		test=np.max(np.abs(dx)) 
		if (test < TOLX):  
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
			return f,x,H,its, 0, ls, msg, dx_norm, incr #FREEALL

	
	msg = "No convergence within %s iterations" %(max_iter,)
	return f,x,H,its,2, ls, msg, dx_norm, incr								#too many iterations in dfpmin				
															#FREEALL


		

	
def maximize(panel, args, callback, mp, comput = None):
	
	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	if comput is None:
		comput = computation.Computation(panel, time.time(), callback, 1e-10) 
	callback(conv = False, done = False)
	f, x, H, its, conv, ls, msg, dx_norm, incr = dfpmin(args,comput, callback, mp, panel)
	callback(f = f, x = x, H = H, its = its, conv = conv, msg = msg, done = True)
	panel.input.args_archive.save(ls.ll.args,True)
	
	return comput, ls, msg, its, conv, dx_norm, incr


def run(panel, args, mp, mp_debug, window, exe_tab, console_output):
	t0=time.time()
	
	comm  = Comm(panel, args, mp, mp_debug, window, exe_tab, console_output, t0)
	comm.callback.print_final(comm.msg, comm.its, comm.incr, comm.f, 1, 'Done', comm.conv, comm.dx_norm, t0, comm.x, comm.ll)
	summary = Summary(comm, t0)

	return summary


class Summary:
	def __init__(self, comm, t0):
		import output
		self.time = time.time() - t0
		self.panel = comm.panel
		self.ll = comm.ll
		self.log_likelihood = comm.ll.LL
		self.coef_params = comm.ll.args.args_v
		self.coef_names = comm.ll.args.names_v
		self.coef_se, self.coef_se_robust = output.sandwich(comm.comput,100)
		self.converged = comm.conv
		self.hessian = comm.comput.H
		self.gradient_vector = comm.comput.g
		self.gradient_matrix = comm.comput.G
		self.count_samp_size_orig = comm.panel.orig_size
		self.count_samp_size_after_filter = comm.panel.NT_before_loss
		self.count_deg_freedom = comm.panel.df
		N, T , k = comm.panel.X.shape
		self.count_ids = N
		self.count_dates = N
		reg_output = comm.callback.channel.output
		self.table = output.RegTableObj(reg_output)
		self.statistics = output.Statistics(comm.ll, comm.panel)
		
		
		
	def print(self, return_string = False):
		t = self.table.table()[0]
		if return_string:
			return t
		print(t)
		return t
		
	def print_df_summary(self, return_string = False):
		t = self.statistics.gen_df_str(self.panel)
		if return_string:
			return t		
		print(t)		
		
	def print_model_summary(self, return_string = False):
		t = self.statistics.gen_mod_fit()
		if return_string:
			return t		
		print(t)	
		
	def print_adf_stat(self, return_string = False):
		t = self.statistics.adf_str()
		if return_string:
			return t		
		print(t)
	
		
		
		

class Comm:
	def __init__(self, panel, args, mp, mp_debug, window, exe_tab, console_output, t0):
		self.mp = mp
		self.listen = None
		self.panel = panel	
		if not mp is None:#This is put here rather than in the next "if not mp" block for efficiency
			self.listen = self.mp.listen(
				[f"maximize.maximize(panel, {list(args)}, callback, mp)"])
			
		import communication as comm
		self.callback = comm.Callback(window,exe_tab,self.panel, console_output, t0)
		self.comput = computation.Computation(panel, t0, gtol = GTOL, tolx = TOLX) 
		self.callback.set_computation(self.comput)
		
		if not mp is None:
			self.start_listening()
		else:
			comput, ls, msg, its, conv, dx_norm, incr = maximize(panel, args, self.callback.generic, mp_debug, self.comput)
			self.msg = msg
			self.f = ls.f
			self.conv = conv
			self.x = ls.x
			self.ll = ls.ll
			self.its = its
			self.incr = incr
			self.dx_norm = dx_norm

	
	def start_listening(self):
		t0 = time.time()
		while not self.listen.done[0]:
			if  time.time()-t0>1:
				self.print()
		self.print()

			
	def print(self):
		if not hasattr(self,'comput'):
			return
		d = self.listen.inbox[0]
		if not 'g' in d:
			return
		(f, msg , its, incr, x, perc,task, 
		 dx_norm, dx, H, hessin, g, alam, rev, msg, conv, constr) = (d['f'], d['msg'], d['its'], d['incr'], 
						d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
						d['H'], d['hessin'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'])
		
		self.ll = logl.LL(x, self.panel, d['constr'])
		self.comput.exec(dx, dx_norm, hessin, H, f, x, g, incr, rev, alam, its, self.ll, False)
		

		self.msg = msg
		self.f = f
		self.conv = conv
		self.x = x
		self.its = its
		self.incr = incr
		self.dx_norm = dx_norm
		
		self.callback.print(msg, its, incr, self.ll, perc , task, dx_norm)





