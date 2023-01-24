#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
import time
import matplotlib
import parallel

PARALLEL_LAYER1 = False
PARALLEL_LAYER2 = True
CALLBACK_ACTIVE = True
	
N_NODES = 10

t0=time.time()

path = os.getcwd().replace('\\', '/')
subpath = os.path.join(path,'mp').replace('\\', '/')

mp = parallel.Parallel(1, path, PARALLEL_LAYER1, CALLBACK_ACTIVE)


mp.exec(["import maximize\n"
		"import parallel as parallel\n"
		f"mp = parallel.Parallel({N_NODES},'{subpath}', {PARALLEL_LAYER2}, {CALLBACK_ACTIVE}, 1)\n" 
		"mp.exec('import loglikelihood as logl\\n'\n"
		"'import maximize', 'init')\n"], 'init')

print(f"parallel: {time.time()-t0}")


import pandas as pd


import output
import main
import options as opt_module
import inspect
import loaddata


def execute(model_string,dataframe, ID=None,T=None,HF=None,instruments=None, console_output=True):
	"""optimizes LL using the optimization procedure in the maximize module"""
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab')
	r=main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,instruments, console_output, mp, PARALLEL_LAYER2)
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
	#	raise RuntimeError(e)
	return dataframe
		
	
options=opt_module.regression_options()
preferences=opt_module.application_preferences()

