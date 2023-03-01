#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption
from pydoc import importfile
import os
path = os.path.dirname(__file__)
panel =  importfile(os.path.join(path,'panel.py'))
model_parser =  importfile(os.path.join(path,'model_parser.py'))
maximize =  importfile(os.path.join(path,'maximize.py'))


import sys
import numpy as np
import warnings
import os
import time

N_NODES = 1
warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string,dataframe, IDs_name, time_name,heteroscedasticity_factors,options,window,
			exe_tab,instruments, console_output, mp):

	"""optimizes LL using the optimization procedure in the maximize module"""
	if not exe_tab is None:
		if exe_tab.isrunning==False:return
	datainput=input_class(dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,instruments)
	if datainput.timevar is None:
		print("No valid time variable defined. This is required")
		return

	summary = doit(datainput,options,mp,options.pqdkm.value,window,exe_tab, console_output)
	
	return summary

class input_class:
	def __init__(self,dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,instruments):
		
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,heteroscedasticity_factors,instruments,options)
		self.descr=model_string
		self.n_nodes = N_NODES
		self.args=None
		if options.arguments.value!="":
			self.args=options.arguments.value
			
def doit(datainput,options,mp,pqdkm,window,exe_tab, console_output):
	print ("Creating panel")
	pnl=panel.panel(datainput,options,pqdkm)			
	
	if not mp is None:
		mp.send_dict({'panel':pnl})
		mp.collect('init')
		s = mp.dict_file.replace('\\','/')
		mp.exec("panel.init()\n", 'panelinit')
	pnl.init()
	if not mp is None:
		mp.collect('panelinit')
	
	if not options.parallel.value:
		mp = None
	summary = maximize.run(pnl, pnl.args.args_init, mp, window, exe_tab, console_output)

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