#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
#import system_main as main
import main
import options as opt_module
import inspect
import numpy as np
import loaddata
import tempstore
from pandas.api.types import is_numeric_dtype
import pandas as pd





def execute(model_string,dataframe, ID=None,T=None,HF=None,join_table=None,instruments=None, console_output=False):
	"""optimizes LL using the optimization procedure in the maximize module"""
	tempstore.test_and_repair()
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab')
	r=main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,join_table,instruments, console_output)
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


def load(fname,sep=None,load_tmp_data=False):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load(fname,load_tmp_data,sep)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	return dataframe

def load_SQL(conn,sql_string,load_tmp_data=True):

	"""Loads data from an SQL server, using sql_string as query"""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),sql_string)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe
	dataframe=main.loaddata.load_SQL(sql_string,conn,load_tmp_data)
	#except RuntimeError as e:
	#	raise RuntimeError(e)
	return dataframe
		
	
options=opt_module.regression_options()
preferences=opt_module.application_preferences()

