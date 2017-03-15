#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import regstats
import regobj
import warnings
import multi_core
import loaddata
import paneltime_functions as ptf
import maximize
import os
warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)





def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, groups_name=None, sort_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=True,add_intercept=True,
            h=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""
	
	r=[]#restrictions
	
		
	print ("Analyzing variables ...")
	X,x_names,Y,y_name,groups,groups_name,W,w_names,has_intercept=ptf.get_variables(dataframe,model_string,groups_name,w_names,add_intercept,sort_name)
	N,cols=X.shape
	if N*(cols**0.5)>200000 and os.cpu_count()>1:#paralell computing will not increase computation time for 'small' data sets
		master=multi_core.master('regprocs','rp')#for paralell computing
	else:
		master=None
		
	model_key=ptf.get_model_key(X, Y, p, d, q, m, k, groups, fixed_random_eff, W)
	session_db,args_d,conv,not_in_use1,not_in_use2 = ptf.load(model_key,loadargs)
	print ("Creating panel")
	panel=regobj.panel(p, d, q, m, k, X, Y, groups,x_names,y_name,groups_name,fixed_random_eff,args_d,W,w_names,master,descr,dataframe,h,has_intercept)
	print ("Maximizing")
	if args_d is None or conv==False or True:
		ll,g,G,H, conv = maximize.maximize(panel,_print=True)
		ptf.save(model_key,session_db,ll.args_d,conv,not_in_use1,not_in_use2) 		
	else:
		ll=panel.LL(panel.args.args)
		g,G=panel.gradient.get(ll,return_G=True)
		H=panel.hessian.get(ll)
		return panel,g,G,H,ll
	return panel,g,G,H,ll


def execute_model(model, p=1, d=0, q=1, m=1, k=1, 
            fixed_random_eff=2, loadargs=True,add_intercept=True,
            h=None
            ):
	return execute(model.dataframe, model.model_string,
	               p,d,q,m,k,model.groups_name,model.descr,fixed_random_eff,model.w_names,loadargs,add_intercept,h)
	

def diagnostics(panel,g,G,H,ll,robustcov_lags=100):
	regstats.diagnostics(panel,g,G,H,100,ll)



class model:
	"""Creates a model object, which contains a *dataframe* (a dictionary of numpy column matrices), a *model_string* 
		(a string specifying the model), *groups* (the name of the grouping variable, if specified) and *w_names* (the name of the 
		custom variance weighting variable, if specified)
		"""
	def __init__(self,X,Y,x_names=None,y_name=None,groups=None,groups_name=None,W=None,w_names=None,filters=None,transforms=None,descr="project_1"):


		dataframe, model_string, w_names, groups_name=ptf.get_data_and_model(X,Y,W,groups,x_names,y_name,w_names,groups_name,filters,transforms)	
		self.dataframe=dataframe
		self.model_string=model_string
		self.w_names=w_names
		self.groups_name=groups_name
		self.descr=descr


def load(fname,sep=None,filters=None,transforms=None):

	"""Loads dataframe from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will try to find it."""
	dataframe=loaddata.load(fname,sep)
	ptf.modify_dataframe(dataframe,transforms,filters)
	print ("The following variables were loaded:")
	return dataframe

def from_matrix(numpy_matrix,headings,filters=None,transforms=None):
	dataframe=dict()
	if type(headings)==str:
		headings=fu.clean(headings,',')
	elif type(headings)!=list:
		raise RuntimeError("Argument 'headings' needs to be either a list or a comma separated string")
	if len(headings)!=numpy_matrix.shape[1]:
		raise RuntimeError("The number of columns in numpy_matrix and the number of headings do not correspond")
	for i in range(len(headings)):
		dataframe[headings[i]]=numpy_matrix[:,i:i+1]
	ptf.modify_dataframe(dataframe,transforms,filters)
	return dataframe
