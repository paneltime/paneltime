#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption

from matplotlib import pyplot  as plt
import numpy as np
import panel
import warnings
import multi_core as mc
import loaddata
import model_parser
import maximize
import tempstore
import os
import time

N_NODES = 1
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
	results_obj = results(datainput,options,mp,options.pqdkm.value,window,exe_tab, console_output)
	if window is None and not mp is None:
		mp.quit()
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
		self.mp=mp
		
		t0=time.time()		
		if not mp is None:
			mp.send_dict({'panel':pnl},
						 command=("panel.init()\n"), wait = False)
		print(f"t:{time.time()-t0}, 3")
		t0=time.time()		
		pnl.init()
		print(f"t:{time.time()-t0}, 4")
		mp.send_dict_by_file_receive()
		t0=time.time()		
		t0 = time.time()
		
		self.ll, self.conv, self.H, self.g, self.G = maximize.run(pnl, pnl.args.args_init.args_v, mp, window, exe_tab, console_output)
		print(f"LL: {self.ll.LL}, time: {time.time()-t0}")
		self.panel=pnl


def mp_check(datainput,window):
	
	modules="""
import maximize
"""	
	if window is None:
		mp = mc.multiprocess(datainput.tempfile,N_NODES,modules)
		return mp
	if window.mc is None:
		window.mc = mc.multiprocess(datainput.tempfile,N_NODES,modules)
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