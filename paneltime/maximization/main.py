#!/usr/bin/env python
# -*- coding: utf-8 -*-



from ..output import stat_functions as stat
from .. import likelihood as logl

from ..output import communication as comm
from ..output import output
from . import init


import numpy as np
import time
import itertools
from queue import Queue
import os

EPS=3.0e-16 
TOLX=(4*EPS) 
GTOL = 1e-5



TEST_ITER = 30



def maximize(panel, args, mp, t0, comm):

  task_name = 'maximization'
  
  gtol = panel.options.tolerance.value

  if mp is None or panel.args.initial_user_defined:
    node = 5
    
    d = maximize_node(panel, args.args_v, gtol, {}, {}, 0, False, False)    
    d['node'] = node
    return d

  tasks = []
  a = get_directions(panel, args, mp.n_slaves)
  for i in range(len(a)):
    tasks.append(
                  f'maximization.maximize_node(panel, {list(a[i])}, {gtol}, inbox, outbox, slave_id, False, True)\n'
                )

  r = mp.exec(tasks, task_name)
  return r



def get_directions(panel, args, n):
  if n == 1:
    return [args.args_v]
  d = args.positions
  size = panel.options.initial_arima_garch_params.value
  pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
  perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
  perm[:,:2] =perm[:,:2]*0.1
  a = np.array([args.args_v for i in range(len(perm))])
  a[:,pos] = perm
  return a


def maximize_node(panel, args, gtol = 1e-5, inbox = {}, outbox = {}, slave_id =0 , nummerical = False, diag_hess = False):
  
  
  import cProfile
  profiler = cProfile.Profile()
  profiler.enable()
  
  res = init.maximize(args, inbox, outbox, panel, gtol, TOLX, nummerical, diag_hess, slave_id)
  
  profiler.disable()
  profiler.print_stats(sort='cumulative')
  
  
  #debug from . import debug
  #debug.save_reg_data(ll, panel)	

  H, G, g, ll = res['H'], res['G'], res['g'], res['ll']

  ll.standardize(panel)
  res['rsq_st'] = stat.goodness_of_fit(ll,True,panel)
  res['rsq'] = stat.goodness_of_fit(ll,True,panel)
  res['var_RE'] = panel.var(ll.e_RE)
  res['var_u'] = panel.var(ll.u)
  return res



def run(panel, args, mp, window, exe_tab, console_output):
  t0=time.time()
  comm  = Comm(panel, args, mp, window, exe_tab, console_output, t0)
  summary = Summary(comm, panel, t0)

  return summary


class Summary:
  def __init__(self, comm, panel, t0):
    self.output = comm.channel.output

    #coefficient statistics:
    self.coef_params = comm.ll.args.args_v
    self.coef_names = comm.ll.args.caption_v
    self.coef_se, self.coef_se_robust = output.sandwich(comm.H, comm.G, comm.g, comm.constr, panel, 100)
    self.table = output.RegTableObj(panel, comm.ll, comm.g, comm.H, comm.G, comm.constr, comm.dx_norm, self.output.model_desc)
    self.coef_tstat = self.table.d['tstat']
    self.coef_tsign = self.table.d['tsign']
    self.coef_codes = self.table.d['sign_codes']
    self.coef_025 = self.table.d['conf_low'] 
    self.coef_0975 = self.table.d['conf_high']


    #other statistics:
    self.time = time.time() - t0
    self.panel = panel
    self.ll = comm.ll
    self.log_likelihood = comm.ll.LL

    self.converged = comm.conv
    self.hessian = comm.H
    self.gradient_vector = comm.g
    self.gradient_matrix = comm.G
    
    self.x = comm.x
    self.count_samp_size_orig = panel.orig_size
    self.count_samp_size_after_filter = panel.NT_before_loss
    self.count_deg_freedom = panel.df
    N, T , k = panel.X.shape
    self.count_ids = N
    self.count_dates = T
    

    self.statistics = output.Statistics(comm.ll, panel)
    self.CI , self.CI_n = self.output.get_CI(comm.constr)

    self.its = comm.its
    self.dx_norm = comm.dx_norm
    self.msg = comm.msg
    self.comm = comm
    self.t0 = t0

  def __str__(self, statistics = True, diagnostics = True, df_accounting = True):
    return self.comm.channel.print_final(self.comm, self.t0,  statistics, diagnostics, df_accounting)
    

  def results(self, return_string = False):
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
    
  def predict(self, signals=None):
    #debug:
    #self.ll.predict(self.panel.W_a[:,-2], self.panel.W_a[:,-1])
    N,T,k = self.panel.W_a.shape
    if signals is None:
      pred = self.ll.predict(self.panel.W_a[:,-1], None)
      return pred
    if not hasattr(signals, '__iter__'):#assumed float
      signals = np.array([signals])
    else:
      signals = np.array(signals)
    if len(signals.shape)>1 or signals.shape[0] != k-1:
      raise RuntimeError("Signals must be a float or a one dimensional vector with the same size as variables assigned to HF argument")
    
    signals = np.append([1],signals)
    pred = self.ll.predict(self.panel.W_a[:,-1], signals.reshape((1,k)))
    return pred
    

class Comm:
  def __init__(self, panel, args, mp, window, exe_tab, console_output, t0):
    self.current_max = None
    self.mp = mp
    self.start_time=t0
    self.panel = panel
    self.channel = comm.get_channel(window,exe_tab,self.panel,console_output)
    d = maximize(panel, args, mp, t0, self)

    self.get(d)


  def get(self, d):
    for attr in d:
      setattr(self, attr, d[attr])  
    self.print_to_channel(self.msg, self.its, self.incr, self.ll,  self.dx_norm)

  def print_to_channel(self, msg, its, incr, ll, dx_norm):
    self.channel.set_output_obj(ll, self, dx_norm)
    self.channel.update(self,its,ll,incr, dx_norm)
    ev = np.abs(np.linalg.eigvals(self.H))**0.5
    try:
      det = np.linalg.det(self.H)
    except:
      det = 'NA'
    if (not self.panel.options.supress_output.value) and self.f!=self.current_max:
      print(f"node: {self.node}, its: {self.its},  LL:{self.f}")
    self.current_max = self.f
