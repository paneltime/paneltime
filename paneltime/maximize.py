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
	a = get_directions(panel, args)
	t = time.time()
	
	d = maximize_node(panel, a[5],0.001, diag_hess=True)
	print(f"f:{d['f']}, its: {d['its']}, t:{time.time()-t}")
	print(d['x'])
	return d

def maximize(panel, args, inbox, outbox, mp, t0):

	task_name = 'maximization'
	callbk = callback.CallBack(inbox, outbox)
	
	a = get_directions(panel, args)
	#a = a[6:8]
	if not mp.is_parallel:
		a = [a[5]]#for debug
	tasks = []
	
	for i in range(len(a)):
		tasks.append(
			f'maximize.maximize_node(panel, {list(a[i])}, 0.001, inbox, outbox, slave_id, False, True)\n'
		)
	evalnodes = EvaluateNodes(mp, len(tasks), t0, panel)
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
	def __init__(self, mp, n, t0, panel):
		self.panel = panel
		self.mp = mp
		self.n_tests = 0
		self.n = n
		self.t = t0
		self.included = list(range(n))
		self.accuracy = self.panel.options.accuracy.value
		

	def get(self, cb):
		f = np.array(get_cb_property(cb, 'f'))
		conv = np.array(get_cb_property(cb, 'conv', 0))>0
		its = np.array(get_cb_property(cb, 'its',0))
		ci = np.array(get_cb_property(cb, 'CI',1e+300))
		ci_anal = np.array(get_cb_property(cb, 'CI_anal',1e+300))
		tpgain = np.array(get_cb_property(cb, 'tpgain',0))
		x = get_cb_property(cb, 'x',0)
		x=np.array(x)
		

		
		terminated = np.array(self.mp.check_state())[:self.n]==0

		flist = np.array(f)
		flist = [i for i in f if not i is None]

		if np.all(terminated):
			citot = ci#np.minimum(ci,1e+150)*ci_anal
			if np.any(conv):
				ctotconv = citot[conv]
				srt = np.argsort(ctotconv)
				#minctot = citot[srt[min((sum(conv),3))-1]]
				minctot = citot[srt[0]]
				fval = max(f[citot<=minctot])
			else:
				fval = max(f)
			ix = list(f).index(fval)
			if True:
				for i in range(len(f)):
					print(f"f:{f[i]},CI:{ci[i]}, x:{cb[i]['x'][:3]}, CI_anal: {ci_anal[i]}, citot:{tpgain[i]}")				
				print(ix)
				print(x[ix])
			return ix, ix

		if len(flist)==0:
			return None, None
		bestix = list(f).index(max(flist))
		return None, bestix
	
	def handle_conv(self, conv, its, terminated, cb, f):
		fdict = get_cb_property(cb, 'fdict', {})
		converged = {}
		for i, c in enumerate(conv):
			if c and not i in converged:
				converged[i] = its[i]
		cur_its = max(converged.values())

		if np.all((its>=cur_its)|terminated):#ensures that the same results are returned every time
			maxf = max(np.array(f)[list(converged.keys())])
			maxix = f.index(maxf)	
			if self.accuracy == 0:
				return maxix
		else:
			return None
		if (len(converged)>1 or (len(converged)&np.all((its>100)|terminated))) and (not maxf is None):
			
			if self.accuracy == 1:
				return maxix
			non_conv = np.arange(len(cb))[(conv==False)|(terminated==False)]
			if len(non_conv):
				cur_its = (its>=cur_its)*cur_its
				f_at_cur_its = np.array([fdict[i][cur_its[i]] for i in range(len(cb))])
				f_nonconv = max(f_at_cur_its[non_conv])
				if f_nonconv-maxf<=1:
					return maxix
			else:
				return maxix 
		
		return None
	
	def quit_procs(self, test_its, cb, terminated, its):
		fdict = get_cb_property(cb, 'fdict', {})
		
		if not np.all((its>=test_its)|terminated):
			return
		n = len(cb)
		quit = 0	
			
		ftest = []
		for i in range(len(fdict)):
			d = fdict[i]
			if not terminated[i]:
				ftest.append(d[test_its])
			else:
				last_it = min((np.sort(list(d.keys()))[-1], test_its))
				ftest.append(d[last_it])
		srt = np.argsort(ftest)
		self.terminated_nodes = srt[:quit]
		self.included = srt[quit:]
		for i in self.terminated_nodes:
			if not terminated[i]:
				self.mp.callback('maximization', {'quit':True}, i)
				terminated[i] == True


def get_directions(panel, args):
	d = args.positions
	size = panel.options.initial_arima_garch_params.value
	pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
	perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
	perm[:,:2] =perm[:,:2]*0.1
	a = np.array([args.args_v for i in range(len(perm))])
	a[:,pos] = perm
	return a


def maximize_node(panel, args, gtol = 1e-5, inbox = {}, outbox = {}, slave_id =0 , nummerical = False, diag_hess = False):

	#have to  completely redesign callback, so that it takes only a dict as argument
	args = np.array(args)
	callbk = callback.CallBack(inbox, outbox)
	comput = computation.Computation(panel, gtol, TOLX, None, nummerical, diag_hess)
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
			self.get(d,True)
		d = self.mp.collect('maximize',0)
		self.get(d, False)


	def get(self, d, prnt = True):
		if not 'g' in d:
			return False
		(self.f, self.its, self.incr, self.x, self.perc,self.task, 
         self.dx_norm, self.dx, self.H, self.G, self.g, self.alam, self.rev, 
		 self.msg, self.conv, self.constr, terminate, self.node) = (
			 d['f'], d['its'], d['incr'], d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'], 
			 d['H'], d['G'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'], d['terminate'], d['node'])

		self.ll = logl.LL(self.x, self.panel, self.constr)
		self.print_to_channel(self.msg, self.its, self.incr, self.ll, self.perc , self.task, self.dx_norm, prnt)
		
	def print_to_channel(self, msg, its, incr, ll, perc , task, dx_norm, prnt):
		if not self.channel.output_set:
			self.channel.set_output_obj(ll, self, dx_norm)
		self.channel.set_progress(perc ,msg ,task=task)
		self.channel.update(self,its,ll,incr, dx_norm)
		ev = np.abs(np.linalg.eigvals(self.H))**0.5
		try:
			det = np.linalg.det(self.H)
		except:
			det = 'NA'
		if self.panel.input.paralell2 and prnt:
			print(f"sid: {self.node}, its:{its}, LL:{ll.LL}, det:{det}, sum pos ev: {sum(ev>0)}, cond number: {ev[0]/(ev[-1]+(ev[-1]==0))}")
		a=0	





