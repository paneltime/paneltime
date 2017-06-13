#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption

import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
import main
import sim_module
import functions as fu


def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, groups_name=None, sort_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=True,direction_testing=True,add_intercept=True,
            h=None,user_constraints=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""
	
	return main.execute(dataframe, model_string, p, d, q, m, k, groups_name, sort_name,
		           descr,
		           fixed_random_eff, w_names, loadargs,direction_testing,add_intercept,
		           h,user_constraints
		           )


def execute_model(model, p=1, d=0, q=1, m=1, k=1, 
            fixed_random_eff=2, loadargs=True,add_intercept=True,
            h=None
            ):
	return main.execute(model.dataframe, model.model_string,
	               p,d,q,m,k,model.groups_name,model.sort_name,model.descr,fixed_random_eff,model.w_names,loadargs,add_intercept,h)
	

def diagnostics(panel,g,G,H,ll,robustcov_lags=100):
	main.regstats.diagnostics(panel,g,G,H,100,ll)



class model:
	"""Creates a model object, which contains a *dataframe* (a dictionary of numpy column matrices), a *model_string* 
		(a string specifying the model), *groups* (the name of the grouping variable, if specified) and *w_names* (the name of the 
		custom variance weighting variable, if specified)
		"""
	def __init__(self,X,Y,x_names=None,y_name=None,groups=None,groups_name=None,W=None,w_names=None,filters=None,transforms=None,descr="project_1",sort_name=None):


		dataframe, model_string, w_names, groups_name=main.ptf.get_data_and_model(X,Y,W,groups,x_names,y_name,w_names,groups_name,filters,transforms)	
		self.dataframe=dataframe
		self.model_string=model_string
		self.w_names=w_names
		self.groups_name=groups_name
		self.descr=descr
		self.sort_name=sort_name


def load(fname,sep=None,filters=None,transforms=None):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	try:
		dataframe=main.loaddata.load(fname,sep)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	main.ptf.modify_dataframe(dataframe,transforms,filters)
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
	main.ptf.modify_dataframe(dataframe,transforms,filters)
	return dataframe


def simulation(N,T,beta,rho=[0.0001],lmbda=[-0.0001],psi=[0.0001],gamma=[00.0001],omega=0.1,mu=1,z=1,residual_sd=0.001,group_sd=0.00001,names=['x','const','Y','ID']):
	return sim_module.simulation(N,T,beta,rho,lmbda,psi,gamma,omega,mu,z,residual_sd,group_sd,names)