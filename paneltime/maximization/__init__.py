#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..output import output


import numpy as np
import time


class Summary:
	def __init__(self, result, panel, t0):
		self.output = output.Output(result, panel, time.time() - t0)
		self.table = output.RegTableObj(panel, result, self.output)
		c = result.ll.args
		self.names = Names(list(c.args_d), list(c.caption_v), list(c.names_v))
		self.count = Counting(panel)
		self.panel = panel
		self.ll = result.ll
		self.stats = self.output.stats #contains clases info and diag with information about the regression and diagnostics
		self.results = Results(panel, result, self.table)
		self.general = General(panel, result, t0, self.output)
		self.random_effects = self.output.random_effects
		
		name_y = panel.input.Y_names[0]
		self.prediction_names = {'Observed':f'Observed {name_y}', 
						   		 'Predicted':f'Predicted {name_y}', 
								 'Fitted':f'Fitted {name_y}'}		

	def __str__(self):
		return '\n\n'.join([
			self.statistics(), 
			self.results_table(), 
			self.diagnostics(), 
			self.accounting()]
			)
	
	def latex(self):
		return self.results_table(fmt='LATEX')
	
	def html(self):
		return self.results_table(fmt='HTML')

	def results_table(self, fmt='CONSOLE'):
		tbl,llength = self.table.table(5,fmt,False,
												show_direction=False,
												show_constraints=False, 
												show_confidence = True)	      
		return tbl

	def statistics(self):
		return self.output.statistics()		

	def diagnostics(self):
		return self.output.diagnostics()

	def accounting(self):
		return self.output.df_accounting()
		
		
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
	def __init__(self, panel, result, table):
		#coefficient statistics:

		self.params = result.x
		self.args = result.ll.args.args_d
		self.se = table.d.get('se_st',None)
		self.coef_se_robust = table.d.get('se_robust',None)
		self.cov = table.d.get('cov',None)
		self.cov_robust = table.d.get('cov_robust',None)
		self.tstat = table.d.get('tstat',None)
		self.tsign = table.d.get('tsign',None)
		self.codes = table.d.get('sign_codes',None)
		self.conf_025 = table.d.get('conf_low' ,None)
		self.conf_0975 = table.d.get('conf_high',None)
		self.codes_def = panel.sign_codes
		self.codes_def_tex = panel.sign_codes_tex
		#NEED TO GO THROUGH HOW RESIDUALS ARE REPORTED 
		#RESIDUALS MUST BE CONSTISTENT WITH RE/fe
		observed = np.asarray(panel.included[3][:, :, 0], dtype=bool)
		residuals = np.asarray(result.ll.e_RE_norm_centered_long, dtype=float).reshape(-1)
		if residuals.size == observed.size:
			residuals = residuals[observed.reshape(-1)]
		self.residuals = residuals
		self.fitted = np.asarray(panel.Y, dtype=float)[observed].reshape(-1) - self.residuals
		self.nobs = int(panel.NT)
		self.df_resid = int(panel.df)


class General:
	def __init__(self, panel, result, t0, output):
		#other statistics:
		self.time = time.time() - t0
		self.t0 = t0
		self.t1 = time.time()
		
		self.panel = panel
		self.ll = result.ll
		self.log_likelihood = result.f

		self.converged = result.conv
		self.hessian = result.H
		self.gradient_vector = result.g
		self.gradient_matrix = result.G
		self.ci, self.ci_n = output.stats.diag.get_CI(result.constr)

		self.its = result.its
		self.dx_norm = result.dx_norm
		self.msg = result.msg
