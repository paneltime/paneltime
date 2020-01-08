#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import output
import panel
import warnings
import multi_core as mc
import loaddata
import model_parser
import maximize
import tempstore
import os
import direction as drctn
import tempstore


warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string,dataframe, IDs_name, time_name,heteroscedasticity_factors,settings):

	"""optimizes LL using the optimization procedure in the maximize module"""

	settings.heteroscedasticity_factors=heteroscedasticity_factors
	datainput=input_class(dataframe,model_string,IDs_name,time_name, settings)
	if settings.loadargs.value==2:
		settings.pqdkm.value=datainput.args_archive.pqdkm
	mp=mp_check(datainput)
	pqdkm=makelist(settings.pqdkm.value)
	results_obj=None
	for i in pqdkm:
		print(f'pqdkm={i}')
		results_obj=results(dataframe,datainput,settings,mp,window,i,results_obj)
	if not mp is None:
		mp.quit()
	return results_obj


def makelist(pqdkm):
	try:
		a=pqdkm[0][0]
		return pqdkm
	except:
		return  [pqdkm]
	

class input_class:
	def __init__(self,dataframe,model_string,IDs_name,time_name, settings,descr):
		
		t=type(settings.user_constraints.value)
		if t!=list and t!=tuple and (not t is None):
			print("Warning: user user_constraints must be a list of tuples. user_constraints are not applied.")	
			
		self.tempfile=tempstore.tempfile_manager()
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,settings)
		self.descr=descr
		if descr==None:
			self.descr=model_string
		self.args_archive=tempstore.args_archive(self.descr, settings.loadargs.value)
		self.args=self.args_archive.args

	
	
class results:
	def __init__(self,dataframe,datainput,settings,mp,window,pqdkm,old_results):
		print ("Creating panel")
		if not old_results is None:
			datainput.args=old_results.ll.args_d	
		pnl=panel.panel(dataframe,datainput,settings,pqdkm)
		direction=drctn.direction(pnl,mp)	
		self.mp=mp
		if not mp is None:
			mp.send_dict_by_file({'panel':pnl})
		self.ll,self.direction,self.printout_obj = maximize.maximize(pnl,direction,mp,pnl.args.args_init,window)	
		self.panel=direction.panel


def mp_check(datainput):
	
	N,k=datainput.X.shape
	mp=None
	if ((N*(k**0.5)>200000 and os.cpu_count()>=2) or os.cpu_count()>=24) or True:#numpy all ready have multiprocessing, so there is no purpose unless you have a lot of processors or the dataset is very big
		modules="""
global cf
global lgl
import calculus_functions as cf
import loglikelihood as lgl
"""
		mp=mc.multiprocess(datainput.tempfile,16,modules,['GARM','GARK','AMAq','AMAp'])
		
	return mp
	
	
