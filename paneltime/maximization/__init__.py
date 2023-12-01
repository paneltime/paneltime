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
    d = maximize.maximize(panel, args, mp, t0)

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
