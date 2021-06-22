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
import direction as drctn
from gui import gui
import communication as comm
import functions as fu


warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string,dataframe, IDs_name, time_name,heteroscedasticity_factors,options,window,exe_tab,join_table,instruments):

	"""optimizes LL using the optimization procedure in the maximize module"""
	if not exe_tab is None:
		if exe_tab.isrunning==False:return
	channel=comm.get_channel(window,exe_tab)
	datainput=input_class(dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments)
	if datainput.timevar is None:
		print("No valid time variable defined. This is required")
		return
	if options.loadARIMA_GARCH.value:
		options.pqdkm.value=datainput.args_archive.pqdkm
	mp,close_mp=mp_check(datainput,window)
	results_obj=pqdkm_iteration(datainput,options,mp,channel)
	if not mp is None and close_mp:
		mp.quit()
	return results_obj
	
def pqdkm_iteration(datainput,options,mp,channel):#allows for a list of different ARIMA options, for example by starting with a more restrictive model
	pqdkm=options.pqdkm.value
	try:
		a=pqdkm[0][0]
	except:
		pqdkm=[pqdkm]
	for i in pqdkm:
		print(f'pqdkm={i}')
		results_obj=results(datainput,options,mp,channel,i)
		if len(pqdkm)>1:
			options.loadargs.value=2
			options.loadARIMA_GARCH.value=True	
	return results_obj
	

class input_class:
	def __init__(self,dataframe,model_string,IDs_name,time_name, options,heteroscedasticity_factors,join_table,instruments):
		
		tempstore.test_and_repair()
		self.tempfile=tempstore.tempfile_manager()
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,heteroscedasticity_factors,instruments,options)
		self.descr=model_string
		self.args_archive=tempstore.args_archive(self.descr, options.loadargs.value)
		self.args=None
		if options.arguments.value!="":
			self.args=options.arguments.value
		self.join_table=join_table
			
		

	
	
class results:
	def __init__(self,datainput,options,mp,channel,pqdkm):
		print ("Creating panel")
		pnl=panel.panel(datainput,options,pqdkm)
		direction=drctn.direction(pnl,mp,channel)	
		self.mp=mp
		if not mp is None:
			mp.send_dict_by_file({'panel':pnl})
		log=[]
		self.ll,self.direction,self.printout_obj = maximize.maximize(pnl,direction,mp,pnl.args.args_init,channel,log=log)
		fu.savevar(log,'log_of_LL_process.csv')
		self.panel=direction.panel


def mp_check(datainput,window):
	modules="""
global cf
global lgl
import calculus_functions as cf
import loglikelihood as lgl
"""	
	if window is None:
		mp=mc.multiprocess(datainput.tempfile,16,modules,['GARM','GARK','AMAq','AMAp'])
		return mp, True
	if window.mc is None:
		window.mc=mc.multiprocess(datainput.tempfile,16,modules,['GARM','GARK','AMAq','AMAp'])
	return window.mc,False
	


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