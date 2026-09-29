#!/usr/bin/env python
# -*- coding: utf-8 -*-



import time
import os
from .output import formatting

import inspect
import warnings

from . import likelihood as logl
from . import main
from . import options as opt_module
from . import info

import numpy as np

import sys

import pandas as pd

from .api import Model, Results
from .options import Effects, FitOptions, OptimizerOptions

def execute(model_string, dataframe, timevar=None, idvar=None, het_factors=None, instruments=None):

	"""Deprecated one-step interface; use ``Model(...).fit()``."""
	warnings.warn(
		'execute() is deprecated; construct Model and call fit().',
		DeprecationWarning, stacklevel=2)
	if not isinstance(dataframe, pd.DataFrame):
		raise ValueError("Input is not a pandas DataFrame. Please provide a valid DataFrame.")
	if dataframe.empty:
		raise ValueError("Input DataFrame is empty. Expected non-empty data.")
	
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window', 'geometry')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab', 'isrunning')

	model = Model(model_string, dataframe, entity=idvar, time=timevar)
	r = model.fit(_legacy_options=options, _het_factors=het_factors, _instruments=instruments)

	return r

def format(summaries, heading, caption, col_headings = [], variable_groups = {}, digits=3, fmt='latex', size = 1, fpath = None):
	"""Prints the results of a set of summaries .\n"""
	s = formatting.format(summaries, fmt, heading, col_headings, variable_groups, digits, size, caption, fpath)

	return s


__version__ = info.version

options=opt_module.create_options(deprecated=True)



