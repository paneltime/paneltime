#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import paneltime_mp as pmp
import time
import os

import inspect

from . import likelihood as logl
from . import main
from . import options as opt_module
from . import info



import numpy as np

import sys

import pandas as pd

import inspect


mp = None

def enable_parallel():
  global mp
  
  mp = pmp.Master(9)
  mp.exec("from paneltime.maximization import maximize as max")
  #this import creates overhead that is not immediately visible, since paneltime_mp is non-blocking. 
  #Paneltime basically needs to be initiated an extra time in the nodes, and paneltime_mp will not
  #start another process before this import has finished. Parallel therefore takes considerably more time
  #than running a single thread. The benefit is that it tries out different directions at once.


def execute(model_string,dataframe, ID=None,T=None,HF=None,instruments=None, console_output=True):

  """Maximizes the likelihood of an ARIMA/GARCH model with random/fixed effects (RE/FE)\n
	model_string: a string on the form 'Y ~ X1 + X2 + X3\n
	dataframe: a dataframe consisting of variables with the names usd in model_string\n
	ID: The group identifier\n
	T: the time identifier\n
	HF: list with names of heteroskedasticity factors (additional regressors in GARCH)\n
	instruments: list with names of instruments
	console_output: if True, GUI output is turned off (GUI output is experimental)

  Note that '++' will add two variables and treat the sum as a single variable
  '+' separates variables
	"""

  window=main.identify_global(inspect.stack()[1][0].f_globals,'window', 'geometry')
  exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab', 'isrunning')

  r = main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,instruments, console_output, mp)

  return r


__version__ = info.version

options=opt_module.create_options()
preferences=opt_module.application_preferences()


