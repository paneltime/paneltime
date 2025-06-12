#!/usr/bin/env python
# -*- coding: utf-8 -*-



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
#multiprocessing module can be enabled here by importing the paneltime_mp package
#for implementation see this version of this repository https://github.com/paneltime/paneltime/commit/65d9f9d08eb3b722526cc708402db73b8f6188bb
#However, this is currently experimental and not recommended for general use.




def execute(model_string,dataframe, timevar = None, idvar = None, het_factors=None, instruments=None):

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

	r = main.execute(model_string, dataframe, timevar, idvar, het_factors, options, window, exe_tab, instruments, True, mp)

	return r


__version__ = info.version

options=opt_module.create_options()
preferences=opt_module.application_preferences()


