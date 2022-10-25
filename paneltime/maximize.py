import numpy as np
import time
import loglikelihood as logl
import computation
import direction
import linesearch
import itertools
#import stat_functions as stat


#This module finds the array of arguments that minimizes some function. The derivative 
#of the function also needs to be supplied. 
#This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by 
#Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'. 
#Cambridge: Cambridge University Press.



EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5

def dfpmin(x, comput, callback, panel, maxiter):
	"""Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
	Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
	its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
	gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
	iter (the number of iterations that were performed), and fret (the minimum value of the
	function). The routine lnsrch is called to perform approximate line minimizations.
	fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """


	x, ll, f, g, hessin, H = comput.calc_init_dir(x)

	its, msg = 0, ''

	for its in range(maxiter):  	#Main loop over the iterations.
		constr = comput.constr

		dx, dx_norm = direction.get(g, x, H, constr, hessin, simple=False)
		ls = linesearch.LineSearch(x, comput, panel)
		ls.lnsrch(x, f, g, dx)	

		dx = ls.x - x
		incr = ls.f - f


		x, f, hessin, H, g, conv = comput.exec(dx, dx_norm,  hessin, H, ls.f, ls.x, ls.g, incr, ls.rev, ls.alam, its, ls.ll)

		err = np.max(np.abs(dx)) < TOLX
		
		terminate = conv or err or its+1==maxiter

		if conv:
			msg = "Convergence on zero gradient; local or global minimum identified"
		elif err:
			msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
		elif terminate:
			msg = "No convergence within %s iterations" %(maxiter,)
			
		callback_function(callback, ls, msg, dx_norm, f, x, H, comput.G, g, hessin, dx, 
						  incr, its, comput.constr, 'linesearch', 
						  panel.arma_dot.perc_calc)			

		if terminate:		
		

			return
			



															
def callback_function(callback, ls, msg, dx_norm, f, x, H, G, g, hessin, dx, incr, its, 
					  constr, task, perc_calc):
	if msg == '':
		msg = ls.msg
		
	callback(msg = msg, dx_norm = dx_norm, f = f, x = x, 
			 H = H, G=G, g = g, hessin = hessin, dx = dx, 
			 incr = incr, rev = ls.rev, alam = ls.alam, 
			 its = its, constr = constr, perc=min(its/100, 1), task = task, 
			 perc_calc = perc_calc)
		
	

def timeit(msg, t0):
	t1=time.time()
	print(f"{msg}:{t1-t0}")
	return t1


def maximize(panel, args, callback, mp, debug_mode, comput = None, start_time=0):
	
	t0=time.time()
	callback(init_time=t0-float(start_time))
	a = get_directions(panel, args)
	tasks = []
	for i in range(len(a)):
		tasks.append(
			f'maximize.maximize_node(panel, {list(a[i])}, callback, nummerical=True, gtol=1e-2, maxiter=200)\n'
			f'node={i}'
		)
	if False:#set to true for parallell/non-debug mode
		paralell = mp.listen(tasks)
		while sum(paralell.done)<len(tasks):
			callback(**paralell.inbox[paralell.last])
			time.sleep(0.01)
		paralell.quit()
		r = paralell.inbox
	else:
		r=[]
		c = CallBackDebug()
		for i in range(1):
			exec(f'maximize_node(panel, {list(a[i])}, callback, nummerical=False, gtol=1e-2)\n')
			#r.append(c.kwargs)
	f = [d['f'] if 'f' in d else -1e+300 for d in r]
	n_max = f.index(max(f))
	d = r[n_max]
	
	d['time used'] = time.time()-t0
	d['last time'] = time.time()
	d['maximization done'] = True
	callback(**d)
		
class CallBackDebug:
	def __init__(self):
		self.kwargs = {}
	def callback(self,**kwargs):
		for k in kwargs:
			self.kwargs[k] = kwargs[k]


def get_directions(panel, args):
	d = args.positions
	size = panel.options.initial_arima_garch_params.value
	pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
	perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)

	perm[:,:2] =perm[:,:2]*0.2
	#perm[:,2:] =perm[:,2:]*0.2
	a = np.array([args.args_v for i in range(len(perm))])
	a[:,pos] = perm
	return a




def maximize_node(panel, args, callback , maxiter = 10000, nummerical = False, 
                  gtol=GTOL):

	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	comput = computation.Computation(panel, gtol, TOLX, nummerical=nummerical)
	if False:
		LL = logl.LL(args, panel, comput.constr)
		LL.standardize(panel)
		beta = stat.OLS(panel,LL.X_st,LL.Y_st,return_e=False)
		args[:len(beta)]=beta.flatten()
	callback(conv = False, done = False, perc=0)
	dfpmin(args,comput, callback, panel, maxiter)




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
		self.coef_se, self.coef_se_robust = output.sandwich(comm,100)
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
		self.its = comm.its



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
		self.start_time=t0
		self.panel = panel	
		if not mp is None:#This is put here rather than in the next "if not mp" block, so that arranging output can be 
							#done simultainiously with calculattions. 
			mp.send_dict({'args':args})
			self.listen = self.mp.listen(
                [f"maximize.maximize(panel, args, callback, mp, False, start_time = {t0})"])

		import communication as comm
		self.callback = comm.Callback(window,exe_tab,self.panel, console_output, t0)
		self.comput = computation.Computation(panel, GTOL, TOLX) 
		self.callback.set_communication(self)

		if not mp is None:
			self.start_listening()
		else:
			maximize(panel, args, self.callback.generic, mp_debug, True, self.comput)
			d = self.callback.kwds
			self.msg = d['msg']
			self.f = d['f']
			self.conv = d['conv']
			self.x = d['x']
			self.ll =  logl.LL(d['x'], self.panel, d['constr'])
			self.its = d['its']
			self.incr = d['incr']
			self.dx_norm = d['dx_norm']
			self.H = d['H']
			self.g = d['g']
			self.G = d['G']
			self.constr = d['constr']
			if 'time used' in d:
				print(f'Time used {d["time used"]}')
				print(f'Time used since finnish {time.time()-d["last time"]}')
			


	def start_listening(self):
		t0 = time.time()
		done = False
		while not self.listen.done[0]:
			if  time.time()-t0>1:
				done = self.print()
		if not done:
			self.print()


	def print(self):
		if not hasattr(self,'comput'):
			return False
		d = self.listen.inbox[0]
		if not 'g' in d:
			return False
		(self.f, self.its, self.incr, self.x, self.perc,self.task, 
         self.dx_norm, self.dx, self.H, self.G, self.g, self.alam, self.rev, 
		 self.msg, self.conv, self.constr) = (
			 d['f'], d['its'], d['incr'], d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
			 d['H'], d['G'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'])

		self.ll = logl.LL(self.x, self.panel, self.constr)
		self.comput.exec(self.dx, self.dx_norm, None, self.H, self.f, self.x,self.g, self.incr, 
						 self.rev, self.alam, self.its, self.ll, False)

		self.callback.print(self.msg, self.its, self.incr, self.ll, self.perc , self.task, self.dx_norm)
		if 'maximization done' in d:
			return True





