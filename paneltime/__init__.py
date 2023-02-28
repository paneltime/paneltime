#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
import time
try:
	import matplotlib
except:
	matplotlib = None
import parallel
import pandas as pd


import output
import main
import options as opt_module
import inspect
import loaddata
import psutil
import signal
from threading import Thread
import traceback

PARALLEL = True

CALLBACK_ACTIVE = True

	
	
N_NODES = 10

t0=time.time()

path = os.getcwd().replace('\\', '/')

mp = parallel.Parallel(N_NODES, path, PARALLEL, CALLBACK_ACTIVE)

mp.exec("import maximize\n", 'init')

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
	
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab')

	r=main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,instruments, console_output, mp)

	return r

def load_json(fname):

	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load_json(fname)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	return dataframe


def load(fname,sep=None):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load(fname,sep)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	return dataframe

def load_SQL(conn,sql_string):

	"""Loads data from an SQL server, using sql_string as query"""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),sql_string)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe
	dataframe=main.loaddata.load_SQL(sql_string,conn)
	#except RuntimeError as e:
	#	raise e
	return dataframe
		
	
options=opt_module.regression_options()
preferences=opt_module.application_preferences()

