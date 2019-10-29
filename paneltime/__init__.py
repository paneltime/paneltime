#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption

import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
#import system_main as main
import main
import sim_module
import functions as fu
import gui
winheading="Paneltime"
winwidth=1000
winheight=500
def execute(dataframe, model_string, IDs_name=None,time_name=None, description=None):

	"""optimizes LL using the optimization procedure in the maximize module"""
	print ("Executing:")
	iconpath=os.path.join(fu.currentdir(),'paneltime.ico')
	w=gui.window(winheading,iconpath,winheight,winwidth)
	w.run(
	    main.execute,
	    (dataframe, model_string, IDs_name, time_name,
	     description,settings,w),
	    close_when_finished=settings.close_when_finished
	)
	r=w.get()
	
	return r

def statistics(results,robustcov_lags=100,correl_vars=None,descriptives_vars=None):
	return main.output.statistics(results,robustcov_lags,correl_vars,descriptives_vars)


def load(fname,sep=None,filters=None,transforms=None,dateformat='%Y-%m-%d',load_tmp_data=False):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	try:
		dataframe=main.loaddata.load(fname,sep,dateformat,load_tmp_data)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	main.model_parser.modify_dataframe(dataframe,transforms,filters)
	print ("The following variables were loaded:"+str(list(dataframe.keys()))[1:-1])
	return dataframe

def load_SQL(conn,sql_string,filters=None,transforms=None,dateformat='%Y-%m-%d',load_tmp_data=False):

	"""Loads data from an SQL server, using sql_string as query"""
	try:
		dataframe=main.loaddata.load_SQL(conn,sql_string,dateformat,load_tmp_data)
	except RuntimeError as e:
		raise RuntimeError(e)
	main.model_parser.modify_dataframe(dataframe,transforms,filters)
	print ("The following variables were loaded:"+str(list(dataframe.keys()))[1:-1])
	return dataframe

def from_matrix(numpy_matrix,headings,filters=None,transforms=None):
	dataframe=dict()
	if type(headings)==str:
		headings=main.fu.clean(headings,',')
	elif type(headings)!=list:
		raise RuntimeError("Argument 'headings' needs to be either a list or a comma separated string")
	if len(headings)!=numpy_matrix.shape[1]:
		raise RuntimeError("The number of columns in numpy_matrix and the number of headings do not correspond")
	for i in range(len(headings)):
		dataframe[headings[i]]=numpy_matrix[:,i:i+1]
	main.model_parser.modify_dataframe(dataframe,transforms,filters)
	return dataframe


def simulation(N,T,beta,rho=[0.0001],lmbda=[-0.0001],psi=[0.0001],gamma=[00.0001],omega=0.1,mu=1,z=1,residual_sd=0.001,ID_sd=0.00001,names=['x','const','Y','ID']):
	return sim_module.simulation(N,T,beta,rho,lmbda,psi,gamma,omega,mu,z,residual_sd,ID_sd,names)	
		
class settings_class:
	def __init__(self):
		self.pqdmk=[1,1,0,1,1]
		
		#No effects    : fixed_random_eff=0
		#Fixed effects : fixed_random_eff=1
		#Random effects: fixed_random_eff=2
		self.group_fixed_random_eff=2
		self.time_fixed_random_eff=2
		self.variance_fixed_random_eff=True
		
		self.heteroscedasticity_factors=None#list of variables that explain heteroskedasticity
		self.loadargs=1 #0: no loading, 1: load arguments, 2: load ARIMA/GARCH orders
		self.add_intercept=True#adds intercept if not all ready supplied (is tested)
		self.h_function=None#use function definition in python syntax
		self.user_constraints=None#dictonary on the form {'<variable1>':<constraint>, ... }
		self.close_when_finished=False#close the window when finnished
		self.tobit_limits=[None,None]#[lower limit, upper limit] (None indicates no limit)
		self.autofit=False

settings=settings_class()