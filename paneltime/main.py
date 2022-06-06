#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import output
import stat_object
import panel
import warnings
import multi_core as mc
import loaddata
import model_parser
import maximize
import tempstore
import os
from paneltimegui import gui
import communication as comm
import functions as fu
import time

N_NODES = 4
warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string,dataframe, IDs_name, time_name,heteroscedasticity_factors,options,window,exe_tab,join_table,instruments, console_output):

	"""optimizes LL using the optimization procedure in the maximize module"""
	if not exe_tab is None:
		if exe_tab.isrunning==False:return
	datainput=input_class(dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments)
	if datainput.timevar is None:
		print("No valid time variable defined. This is required")
		return
	mp = mp_check(datainput,window)
	results_obj=pqdkm_iteration(datainput,options,mp,window,exe_tab, console_output)
	return results_obj
	
def pqdkm_iteration(datainput,options,mp,window,exe_tab, console_output):#allows for a list of different ARIMA options, for example by starting with a more restrictive model
	pqdkm=options.pqdkm.value
	try:
		a=pqdkm[0][0]
	except:
		pqdkm=[pqdkm]
	for i in pqdkm:
		print(f'pqdkm={i}')
		results_obj=results(datainput,options,mp,i,window,exe_tab, console_output)
		if len(pqdkm)>1:
			options.loadargs.value=2
	return results_obj
	

class input_class:
	def __init__(self,dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments):
		
		tempstore.test_and_repair()
		self.tempfile=tempstore.TempfileManager()
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,heteroscedasticity_factors,instruments,options)
		self.descr=model_string
		self.n_nodes = N_NODES
		self.args_archive=tempstore.args_archive(self.descr, options.loadargs.value)
		self.args=None
		if options.arguments.value!="":
			self.args=options.arguments.value
		self.join_table=join_table
			
		

	
	
class results:
	def __init__(self,datainput,options,mp,pqdkm,window,exe_tab, console_output):
		print ("Creating panel")
		pnl=panel.panel(datainput,options,pqdkm)
		callback = comm.callback(window,exe_tab,pnl, console_output)
		self.mp=mp
		if not mp is None:
			mp.send_dict({'panel':pnl},
						 command=("panel.ARMA_init()\n"))
		pnl.ARMA_init()
		t0 = time.time()
		self.ll, self.comput, self.conv = maximize.maximize(pnl, pnl.args.args_init.args_v, callback)
		print(f"LL: {self.ll.LL}, time: {time.time()-t0}")
		self.panel=pnl


def mp_check(datainput,window):
	return None#Multithreading disabled. Found it did not raise performance, much compared with the disadvantage of the randomness it adds to the results
	modules="""
import maximize_num
"""	
	if window is None:
		mp=mc.multiprocess(datainput.tempfile,N_NODES,modules, 'computation')
		return mp
	if window.mc is None:
		window.mc=mc.multiprocess(datainput.tempfile,N_NODES,modules)
	return window.mc
	


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