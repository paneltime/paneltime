#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..output import communication as comm
from ..output import output
from . import maximize


import numpy as np
import time


def go(panel, args, mp, window, exe_tab, console_output):
	t0=time.time()
	comm  = Comm(panel, args, mp, window, exe_tab, console_output, t0)
	summary = Summary(comm, panel, t0)

	return summary


class Summary:
	def __init__(self, comm, panel, t0):
		self.output = output.Output(comm.ll, panel, comm.dx_norm)
		self.output.update(comm.its, comm.ll, 0, comm.dx_norm, time.time()-t0, comm.conv, comm.msg)
		self.table = output.RegTableObj(panel, comm.ll, comm.g, comm.H, comm.G, comm.constr, comm.dx_norm, self.output.model_desc)
		c = comm.ll.args
		self.names = Names(list(c.args_d), list(c.caption_v), list(c.names_v) )
		self.count = Counting(panel)
		self.panel = panel
		self.ll = comm.ll
		self.results = Results(panel, comm, self.table)
		self.general = General(panel, comm, t0, self.output)
		self.random_effects = RandomEffects(panel, comm.ll)
		
		name_y = panel.input.Y_names[0]
		self.prediction_names = {'Observed':f'Observed {name_y}', 
						   		 'Predicted':f'Predicted {name_y}', 
								 'Fitted':f'Fitted {name_y}'}		

	def __str__(self, statistics = True, diagnostics = True, df_accounting = True):
		return '\n\n'.join([
			self.statistics(), 
			self.results_table(), 
			self.diagnostics(), 
			self.accounting()]
			)

	def results_table(self):
		tbl,llength = self.table.table(5,'(','CONSOLE',False,
																			show_direction=False,
																			show_constraints=False, 
																			show_confidence = True)	      
		return tbl

	def statistics(self):
		return self.output.statistics()		

	def diagnostics(self):
		return self.output.diagnostics(self.general.comm.constr)

	def accounting(self):
		return self.output.df_accounting(self.general.panel)
		
		
	def predict(self, signals=None):
		#debug:
		#self.ll.predict(self.panel.W_a[:,-2], self.panel.W_a[:,-1], self.panel)
		N,T,k = self.panel.W_a.shape
		if signals is None:
			pred = self.ll.predict(self.panel)
			return pred
		if not hasattr(signals, '__iter__'):#assumed float
			signals = np.array([signals])
		else:
			signals = np.array(signals)
		if len(signals.shape)>1 or signals.shape[0] != k-1:
			raise RuntimeError("Signals must be a float or a one dimensional vector with the same size as variables assigned to HF argument")
		
		signals = np.append([1],signals)
		pred = self.ll.predict(self.panel.W_a[:,-1], signals.reshape((1,k)), self.panel)
		return pred
		
class Names:
	def __init__(self, groups, captions, varnames):
		self.groups = groups
		self.captions = captions
		self.varnames = varnames


class Counting:
	def __init__(self, panel):
		N, T , k = panel.X.shape

		self.samp_size_orig = panel.orig_size
		self.samp_size_after_filter = panel.NT_before_loss
		self.deg_freedom = panel.df
		self.groups = N
		self.dates = T
		self.variables = k

class Results:
	def __init__(self, panel, comm, table):
		#coefficient statistics:

		self.params = comm.ll.args.args_v
		self.args = comm.ll.args.args_d
		self.se, self.coef_se_robust = output.sandwich(comm.H, comm.G, comm.g, comm.constr, panel, 100)
		self.tstat = table.d.get('tstat',None)
		self.tsign = table.d.get('tsign',None)
		self.codes = table.d.get('sign_codes',None)
		self.conf_025 = table.d.get('conf_low' ,None)
		self.conf_0975 = table.d.get('conf_high',None)


class General:
	def __init__(self, panel, comm, t0, output):	
		#other statistics:
		self.comm = comm
		
		self.time = time.time() - t0
		self.t0 = t0
		self.t1 = time.time()
		
		self.panel = panel
		self.ll = comm.ll
		self.log_likelihood = comm.ll.LL

		self.converged = comm.conv
		self.hessian = comm.H
		self.gradient_vector = comm.g
		self.gradient_matrix = comm.G
		self.ci , self.ci_n = output.get_CI(comm.constr)

		self.its = comm.its
		self.dx_norm = comm.dx_norm
		self.msg = comm.msg



class RandomEffects:
	def __init__(self, panel, ll):
		self.residuals_i = None
		self.residuals_t = None

		if (panel.options.fixed_random_time_eff + 
				panel.options.fixed_random_group_eff) == 0:
			return
		
		i, t = self.ll.get_re(panel)
		self.residuals_i = None
		self.residuals_t = None

		self.std_i = np.std(i, ddof=1)
		self.std_t = np.std(t, ddof=1)

class Comm:
	def __init__(self, panel, args, mp, window, exe_tab, console_output, t0):
		self.current_max = None
		self.mp = mp
		self.start_time=t0
		self.panel = panel
		self.channel = comm.get_channel(window,exe_tab,self.panel,console_output)
		d = maximize.maximize(panel, args, mp, t0)

		self.get(d)


	def get(self, d):
		for attr in d:
			setattr(self, attr, d[attr])  
		self.print_to_channel(self.msg, self.its, self.incr, self.ll,  self.dx_norm, self.conv)

	def print_to_channel(self, msg, its, incr, ll, dx_norm, conv):
		self.channel.set_output_obj(ll, self, dx_norm)
		self.channel.update(self,its,ll,incr, dx_norm, conv, msg)
		ev = np.abs(np.linalg.eigvals(self.H))**0.5
		try:
			det = np.linalg.det(self.H)
		except:
			det = 'NA'
		if (not self.panel.options.supress_output) and self.f!=self.current_max:
			print(f"node: {self.node}, its: {self.its},  LL:{self.f}")
		self.current_max = self.f
