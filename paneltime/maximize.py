import numpy as np
import time
import loglikelihood as logl
import computation
import itertools
import callback

EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5





def maximize(panel, args, callback, mp, debug_mode, start_time=0):
	
	t0=time.time()

	a = get_directions(panel, args)
	tasks = []
	maxiter = 10
	for i in range(len(a)):
		tasks.append(
			f'callback.callback(node={i})\n'
			f'maximize.maximize_node(panel, {list(a[i])}, callback, nummerical=False, gtol=1e-5, maxiter={maxiter})\n'
		)

	if True:#set to true for parallell/non-debug mode
		mp.wait_untill_done()
		mp.exec(tasks, 'maximization')
		f_maxiter = None
		while f_maxiter is None:
			time.sleep(0.01)
			cb = mp.callback('maximization')
			fltr = [('f_maxiter' in d) for d in cb]
			if sum(fltr)==len(tasks):
				unnassigned = mp.cpu_count - len(tasks)
				if np.all([not d['f_maxiter'] is None for d in cb[:-unnassigned]]):
					f_maxiter = [d['f_maxiter'] for d in cb[:-unnassigned]]
					max_node = f_maxiter.index(max(f_maxiter))
		for i in range(len(tasks)):
			if i == max_node:
				_ = mp.callback('maximization', {'quit':True}, i)
		while True:
			d = mp.callback('maximization', s_id = max_node)
			callback.callback(**d)
			if d['terminated']:
				break
		callback.callback(**d)
	else:
		exec(f'maximize_node(panel, {list(a[2])}, callback, nummerical=False, maxiter={maxiter}, gtol=1e-2)\n')
		callback.callback(node = 0)
			
	callback.callback(time_used=time.time()-t0, last_time = time.time(), maximization_done = True)

	

def get_directions(panel, args):
	d = args.positions
	size = panel.options.initial_arima_garch_params.value
	pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
	perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
	perm[:,:2] =perm[:,:2]*0.2
	a = np.array([args.args_v for i in range(len(perm))])
	a[:,pos] = perm
	return a


def maximize_node(panel, args, callback , maxiter = 10000, nummerical = False, 
                  gtol=GTOL):

	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	comput = computation.Computation(panel, gtol, TOLX, callback.callback, nummerical=nummerical)
	if False:
		LL = logl.LL(args, panel, comput.constr)
		LL.standardize(panel)
		beta = stat.OLS(panel,LL.X_st,LL.Y_st,return_e=False)
		args[:len(beta)]=beta.flatten()
	callback.callback(quit=False, conv=False, perc=0)
	try:
		dfpmax(args,comput, callback, panel, maxiter)
	except RuntimeError as e:
		if str(e)!='Quitting on demand':
			raise RuntimeError(e)




def run(panel, args, mp, mp_debug, window, exe_tab, console_output):
	t0=time.time()

	comm  = Comm(panel, args, mp, mp_debug, window, exe_tab, console_output, t0)
	comm.callback.print_final(comm.msg, comm.its, comm.incr, comm.f, 1, 'Done', comm.conv, comm.dx_norm, t0, comm.x, comm.ll, comm.node)
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
			self.mp.send_dict({'args':args})
			self.mp.wait_untill_done()
			self.mp.exec(
                [f"maximize.maximize(panel, args, callback, mp, False, start_time = {t0})"], 'maximize')

		import communication as comm
		self.callback = callback.CallBack(comm.get_channel(window,exe_tab,self.panel,console_output))
		self.callback.set_communication(self)
		self.comput = computation.Computation(panel, GTOL, TOLX, self.callback.callback) 
		if not mp is None:
			self.start_listening()
		else:
			maximize(panel, args, self.callback, mp_debug, True)
			d = self.callback.outbox
			self.msg = d['msg']
			self.f = d['f']
			self.conv = d['conv']
			self.node = d['node']
			self.x = d['x']
			self.ll =  logl.LL(d['x'], self.panel, d['constr'])
			self.its = d['its']
			self.incr = d['incr']
			self.dx_norm = d['dx_norm']
			self.dx = d['dx']
			self.H = d['H']
			self.g = d['g']
			self.G = d['G']
			self.rev = d['rev']
			self.alam = d['alam']
			self.constr = d['constr']
			self.comput.exec(self.dx, self.dx_norm, None, self.H, self.f, self.x,self.g, self.incr, 
								 self.rev, self.alam, self.its, self.ll, False)			
			if 'time_used' in d:
				print(f'Time used {d["time_used"]}')
				print(f'Time used since finnish {time.time()-d["last_time"]}')
			


	def start_listening(self):
		t0 = time.time()
		done = False
		while self.mp.any_alive():
			if  time.time()-t0>0.5:
				self.print(self.mp.callback('maximize'))
				t0 = time.time()
		self.print( self.mp.callback('maximize'))

	def print(self, d):
		if not hasattr(self,'comput'):
			return False
		if not 'g' in d:
			return False
		(self.f, self.its, self.incr, self.x, self.perc,self.task, 
         self.dx_norm, self.dx, self.H, self.G, self.g, self.alam, self.rev, 
		 self.msg, self.conv, self.constr, self.node, terminated) = (
			 d['f'], d['its'], d['incr'], d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
			 d['H'], d['G'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'], d['node'], d['terminated'])

		self.ll = logl.LL(self.x, self.panel, self.constr)
		self.comput.exec(self.dx, self.dx_norm, None, self.H, self.f, self.x,self.g, self.incr, 
						 self.rev, self.alam, self.its, self.ll, False)
		
		self.callback.print(self.msg, self.its, self.incr, self.ll, self.perc , self.task, self.dx_norm)





