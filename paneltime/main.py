#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import panel
import warnings
import multi_core as mc
import model_parser
import maximize
import tempstore
import os
import time

N_NODES = 1
warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string,dataframe, IDs_name, time_name,heteroscedasticity_factors,options,window,
			exe_tab,join_table,instruments, console_output, mp, mp_debug):

	"""optimizes LL using the optimization procedure in the maximize module"""
	if not exe_tab is None:
		if exe_tab.isrunning==False:return
	datainput=input_class(dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments)
	if datainput.timevar is None:
		print("No valid time variable defined. This is required")
		return

	summary = doit(datainput,options,mp, mp_debug,options.pqdkm.value,window,exe_tab, console_output)
	
	return summary

class input_class:
	def __init__(self,dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments):
		
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,heteroscedasticity_factors,instruments,options)
		self.descr=model_string
		self.n_nodes = N_NODES
		self.args_archive=tempstore.args_archive(self.descr, options.loadargs.value)
		self.args=None
		if options.arguments.value!="":
			self.args=options.arguments.value
		self.join_table=join_table
			
def doit(datainput,options,mp, mp_debug,pqdkm,window,exe_tab, console_output):
	print ("Creating panel")
	pnl=panel.panel(datainput,options,pqdkm)			

	
	
	if not mp is None:
		command = (
			"panel.init()\n"
			"mp.send_dict({'panel':panel}, command='panel.init()\\n' , cleanup = False)\n"
		)
		
		mp.send_dict({'panel':pnl},
					 command = command, cleanup = False)
	else:
		mp_debug.send_dict({'panel':pnl},
					 command = 'panel.init()', cleanup = False)
		

	pnl.init()
	if not options.multi_core.value:
		mp = None
	summary = maximize.run(pnl, pnl.args.args_init, mp, mp_debug, window, exe_tab, console_output)
	
	return summary



def indentify_dataset(glob,source):
	try:
		window=glob['window']
		datasets=window.right_tabs.data_tree.datasets
		for i in datasets:
			data_source=' '.join(datasets[i].source.split())
			editor_source=' '.join(source.split())
			if data_source==editor_source:
				return datasets[i]
	except:
		return False
			

		
def identify_global(globals,name):
	try:
		variable=globals[name]
	except:
		variable=None	
	return variable