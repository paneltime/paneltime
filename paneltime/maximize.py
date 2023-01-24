import numpy as np
import time
import loglikelihood as logl
import computation
import itertools
import callback
import dfpmax
import output
from queue import Queue
import communication as comm
import output

EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5


TEST_ITER = 30

def maximize_single(panel, args):#for debugging
	args = args.args_v
	t = time.time()
	d = maximize_node(panel, args,0.01)
	print(f"f:{d['f']}, its: {d['its']}, t:{time.time()-t}")
	print(d['x'])
	return d

def maximize(panel, args, inbox, outbox, mp, t0):

	task_name = 'maximization'
	callbk = callback.CallBack(inbox, outbox)
	
	a = get_directions(panel, args)
	#a = a[6:8]
	if not mp.is_parallel:
		a = [a[4]]#for debug
	tasks = []
	
	for i in range(len(a)):
		tasks.append(
			f'maximize.maximize_node(panel, {list(a[i])}, 0.001, inbox, outbox, slave_id, False)\n'
		)
	evalnodes = EvaluateNodes(mp, len(tasks), t0)
	mp.exec(tasks, task_name)
	
	while True:
		cb = mp.callback(task_name)	
		if mp.callback_active:
			maxidx, bestix = evalnodes.get(cb[:len(tasks)])
		else:
			while sum(mp.check_state())>0:
				time.sleep(0.1)
			maxidx, bestix = get_final_res(mp, tasks, task_name)
			
		if not maxidx is None:
			break
		if not bestix is None:
			cb[bestix]['node'] = bestix
			callbk.callback(**cb[bestix])
	cb_max = mp.collect(task_name, maxidx)
	cb_max['node'] = maxidx
	return cb_max

def get_cb_property(cb, kw, nonevalue = None):
	values = [d[kw] if kw in d else nonevalue for d in cb]
	return values
	
def get_final_res(mp, tasks, task_name):
	res = mp.collect(task_name)[:len(tasks)]
	f = get_cb_property(res, 'f', -1e+300)
	maxidx = f.index(max(f))
	return maxidx, maxidx

class EvaluateNodes:
	def __init__(self, mp, n, t0):
		self.mp = mp
		self.n_tests = 0
		self.n = n
		self.t = t0
		self.included = list(range(n))
		
		
	def get(self, cb):
		f = get_cb_property(cb, 'f')
		conv = np.array(get_cb_property(cb, 'conv', 0))>0
		its = np.array(get_cb_property(cb, 'its',0))
		k = int(self.n_tests>0)
		test_its = [20, 40][k]
		collect = [5, 1][k]	
		fdict = get_cb_property(cb, 'fdict', {})
		flist = [i for i in f if not i is None]
		terminated = np.array(self.mp.check_state())[:self.n]==0
		if self.n==1 and terminated[0] and False:#debug
			return 0, 0
		if np.all((its>=test_its)+terminated):
			self.n_tests += 1
			ftest = []
			for i,d in enumerate(fdict):
				if not terminated[i]:
					ftest.append(d[test_its])
				else:
					last_it = min((np.sort(list(d.keys()))[-1], test_its))
					ftest.append(d[last_it])
			srt = np.argsort(ftest)
			for i in srt[:-collect]:
				if not terminated[i]:
					self.mp.callback('maximization', {'quit':True}, i)

			if self.n_tests>0 and np.any(conv[-collect:]):
				first = max(np.array(f)[conv])
				max_idx = f.index(first)
				return max_idx, max_idx			
		if len(flist)==0:
			bestix = None
		else:
			bestix = f.index(max(flist))
	
		if np.all(terminated):
			return bestix, bestix
		return None, bestix
	


def get_directions(panel, args):
	d = args.positions
	size = panel.options.initial_arima_garch_params.value
	pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
	perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
	perm[:,:2] =perm[:,:2]*0.1
	a = np.array([args.args_v for i in range(len(perm))])
	a[:,pos] = perm
	return a


def maximize_node(panel, args, gtol = 1e-5, inbox = {}, outbox = {}, slave_id =0 , nummerical = False):

	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	callbk = callback.CallBack(inbox, outbox)
	comput = computation.Computation(panel, gtol, TOLX, None, nummerical)
	callbk.callback(quit=False, conv=False, perc=0)
	res = dfpmax.dfpmax(args,comput, callbk, panel, slave_id)
	return res





def run(panel, args, mp, window, exe_tab, console_output):
	t0=time.time()
	comm  = Comm(panel, args, mp, window, exe_tab, console_output, t0)
	comm.channel.print_final(comm.msg, comm.ll.LL, comm.conv, t0, comm.ll.args.args_v, comm.its, comm.node)
	summary = Summary(comm, t0)

	return summary


class Summary:
	def __init__(self, comm, t0):
		self.time = time.time() - t0
		self.panel = comm.panel
		self.ll = comm.ll
		self.log_likelihood = comm.ll.LL
		self.coef_params = comm.ll.args.args_v
		self.coef_names = comm.ll.args.names_v
		self.coef_se, self.coef_se_robust = output.sandwich(comm,100)
		self.converged = comm.conv
		self.hessian = comm.H
		self.gradient_vector = comm.g
		self.gradient_matrix = comm.G
		self.count_samp_size_orig = comm.panel.orig_size
		self.count_samp_size_after_filter = comm.panel.NT_before_loss
		self.count_deg_freedom = comm.panel.df
		N, T , k = comm.panel.X.shape
		self.count_ids = N
		self.count_dates = N
		reg_output = comm.channel.output
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
	def __init__(self, panel, args, mp, window, exe_tab, console_output, t0):
		self.mp = mp
		self.start_time=t0
		self.panel = panel
		self.mp.send_dict({'args':args, 't0':t0})
		a=0
		
		self.mp.exec(
			[f"maximize.maximize(panel, args, inbox, outbox, mp, t0)"], 'maximize')

		self.channel = comm.get_channel(window,exe_tab,self.panel,console_output)
		self.res = self.listen()


	def listen(self):
		hasprinted = False
		while True:
			t = time.time()
			count = self.mp.count_alive()
			if not count:
				break
			d = self.mp.callback('maximize')[0]
			
			self.get(d)
		d = self.mp.collect('maximize',0)
		self.get(d, False)
		print(f"listen time:{time.time()-self.start_time}")

	def get(self, d, prnt = True):
		if not 'g' in d:
			return False
		(self.f, self.its, self.incr, self.x, self.perc,self.task, 
         self.dx_norm, self.dx, self.H, self.G, self.g, self.alam, self.rev, 
		 self.msg, self.conv, self.constr, terminate, self.node) = (
			 d['f'], d['its'], d['incr'], d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
			 d['H'], d['G'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'], d['terminate'], d['node'])

		self.ll = logl.LL(self.x, self.panel, self.constr)
		if prnt:
			self.print_to_channel(self.msg, self.its, self.incr, self.ll, self.perc , self.task, self.dx_norm)
		
	def print_to_channel(self, msg, its, incr, ll, perc , task, dx_norm):
		if not self.channel.output_set:
			self.channel.set_output_obj(ll, self, dx_norm)
		self.channel.set_progress(perc ,msg ,task=task)
		self.channel.update(self,its,ll,incr, dx_norm)
		ev = np.linalg.eigvals(self.H)
		try:
			det = np.linalg.det(self.H)
		except:
			det = 'NA'
		if self.panel.input.paralell2:
			print(f"sid: {self.node}, its:{its}, LL:{ll.LL}, det:{det}, sum pos ev: {sum(ev>0)}, cond number: {ev[0]/(ev[-1]+(ev[-1]==0))}")
		a=0	





