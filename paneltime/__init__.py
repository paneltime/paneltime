#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo:

#Task 1:
#The main obstacle for efficient paralell computing is ""mp.exec("from paneltime import engine")" below. Because it needs to be run after all other imports, paneltime must be
#importe two times, one for the main thread and a second for the slaves. 

#The solution is to craeate a new pip package called paneltime_core, which the main distribution depends on. Slaves can then be initiated at the beginning, importing a publicly
#available package. 

#In addition, perhaps the parallel package should be turned into an idependent pip-package, paneltime_parallel. 

#With this setup, the paneltime_engine can be loaded simultainously by the main thread and the nodes. 

#Task 2:
#Have the baseline thread (zero starting value for all ARMA/GARCH parameters) in the main thread. 



from .engine import parallel as p
import time
import os
mp = None
import inspect

from .engine import likelihood as logl
from .engine import main
from .engine import options as opt_module
from . import info



import numpy as np

import sys

import pandas as pd

import inspect


def enable_parallel():
  stack = inspect.stack()
  
  global mp
  N_NODES = 10

  t0=time.time()

  #temporary debug output is saved here:

  mp = p.Master(N_NODES, os.path.dirname(__file__))
  engine_path = os.path.join(os.path.dirname(__file__),'engine')
  mp.exec("from paneltime import engine")
  
  print(f"parallel: {time.time()-t0}")

def execute(model_string,dataframe, ID=None,T=None,HF=None,instruments=None, console_output=True):

  """Maximizes the likelihood of an ARIMA/GARCH model with random/fixed effects (RE/FE)\n
	model_string: a string on the form 'Y ~ X1 + X2 + X3\n
	dataframe: a dataframe consisting of variables with the names usd in model_string, ID, T, HF and instruments\n
	ID: The group identifier\n
	T: the time identifier\n
	HF: list with names of heteroskedasticity factors (additional regressors in GARCH)\n
	instruments: list with names of instruments
	console_output: if True, GUI output is turned off (GUI output is experimental)
	"""
  if options.parallel.value:
    enable_parallel()
  
  window=main.identify_global(inspect.stack()[1][0].f_globals,'window', 'geometry')
  exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab', 'isrunning')

  r = main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,instruments, console_output, mp)

  return r


__version__ = info.version

options=opt_module.regression_options()
preferences=opt_module.application_preferences()


