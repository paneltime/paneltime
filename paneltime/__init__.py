#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
#import system_main as main
import main
import sim_module
import functions as fu
from gui import gui
import options as opt_module
import inspect
import numpy as np

#Todo: check that if works for no id and date variable
#add argument for null model (default: Y~Intercept)


def start():
	"""Starts the GUI"""
	window=gui.window()
	window.mainloop() 

def execute(model_string,dataframe, ID=None,T=None,HF=None):
	"""optimizes LL using the optimization procedure in the maximize module"""
	
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab')
	r=main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab)
	return r

def statistics(results,correl_vars=None,descriptives_vars=None,name=None):
	return main.output.statistics(results,correl_vars,descriptives_vars,name)

def load_json(fname):

	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load_json(fname)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	main.model_parser.modify_dataframe(dataframe)
	return dataframe


def load(fname,sep=None,dateformat='%Y-%m-%d',load_tmp_data=False):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load(fname,sep,dateformat,load_tmp_data)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	main.model_parser.modify_dataframe(dataframe)
	return dataframe

def load_SQL(conn,sql_string,dateformat='%Y-%m-%d',load_tmp_data=True):

	"""Loads data from an SQL server, using sql_string as query"""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),sql_string)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe
	dataframe=main.loaddata.load_SQL(conn,sql_string,dateformat,load_tmp_data)
	#except RuntimeError as e:
	#	raise RuntimeError(e)
	main.model_parser.modify_dataframe(dataframe)
	return dataframe

def filter_data(data_filters,dataset):
	d,n=main.model_parser.filter_data(data_filters,dataset)
	return d

def edit_data(edit_script,dataset):
	exec(edit_script,dataset,dataset)
	for i in list(dataset.keys()):
		if not type(dataset[i])==np.ndarray:
			dataset.pop(i)
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	if hasattr(dataset,'datasets') and (not window is None):
		dataset.datasets.make_tree(dataset.name,window.right_tabs.data_tree)	
		
options=opt_module.regression_options()
preferences=opt_module.application_preferences()

