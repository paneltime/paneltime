#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import init


import numpy as np
import time
import itertools
from queue import Queue
import os


EPS=3.0e-16 
TOLX=(4*EPS) 



def maximize(panel, args, mp, t0):


  gtol = panel.options.tolerance.value

  if mp is None or panel.args.initial_user_defined:
    d = maximize_node(panel, args.args_v, gtol, 0, False, False)    
    return d

  tasks = []
  a = get_directions(panel, args, mp.n_slaves)
  for i in range(len(a)):
    tasks.append(
                  f'max.maximize_node(panel, {list(a[i])}, {gtol}, {i}, False, True)\n'
                )
    
  mp.eval(tasks)
  r_base = maximize_node(panel, args.args_v, gtol, len(a), False, False)  
  res = mp.collect()
  res[len(a)] = r_base
  f = [res[k]['f'] for k in res]
  r = res[list(res.keys())[f.index(max(f))]]
  return r



def get_directions(panel, args, n):
  if n == 1:
    return [args.args_v]
  d = args.positions
  size = panel.options.initial_arima_garch_params.value
  pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
  perm = np.array(list(itertools.product([-1,0, 1], repeat=len(pos))), dtype=float)
  z = np.nonzero(np.sum(perm**2,1)==0)[0][0]
  perm = perm[np.arange(len(perm))!=z]
  perm[:,:] =perm[:,:]*0.01
  perm = perm[:-1]
  a = np.array([args.args_v for i in range(len(perm))])
  a[:,pos] = perm
  return a


def maximize_node(panel, args, gtol = 1e-5, slave_id =0 , nummerical = False, diag_hess = False):
  
  

  
  res = init.maximize(args, panel, gtol, TOLX, nummerical, diag_hess, slave_id)
  


  H, G, g, ll = res['H'], res['G'], res['g'], res['ll']

  res['node'] = slave_id

  return res


