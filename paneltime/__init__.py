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
import options


window=None
def start():
	"""Starts the GUI"""
	global window
	window=gui.window(environment=globals())
	window.mainloop() 

def execute(model_string,dataframe, ID=None,T=None,HF=None):
	"""optimizes LL using the optimization procedure in the maximize module"""
	r=main.execute(model_string,dataframe,ID, T,options)
	return r

def statistics(results,correl_vars=None,descriptives_vars=None):
	return main.output.statistics(results,correl_vars,descriptives_vars)


def load(fname,sep=None,filters=None,transforms=None,dateformat='%Y-%m-%d',load_tmp_data=False):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	try:
		dataframe=main.loaddata.load(fname,sep,dateformat,load_tmp_data)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	main.model_parser.modify_dataframe(dataframe,transforms,filters)
	return dataframe

def load_SQL(conn,sql_string,filters=None,transforms=None,dateformat='%Y-%m-%d',load_tmp_data=False):

	"""Loads data from an SQL server, using sql_string as query"""
	try:
		dataframe=main.loaddata.load_SQL(conn,sql_string,dateformat,load_tmp_data)
	except RuntimeError as e:
		raise RuntimeError(e)
	main.model_parser.modify_dataframe(dataframe,transforms,filters)
	return dataframe
		
options=options.options()