#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv



import numpy as np
import output
import system_panel as panel
import warnings
import multi_core as mc
import loaddata
import model_parser
import maximize
import tempstore
import os
import system_direction as drctn


warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, IDs_name=None, time_name=None,
            descr=None,
            group_fixed_random_eff=2, time_fixed_eff=True, heteroscedasticity_factors=None, loadargs=1,add_intercept=True,
            h=None,user_constraints=None,window=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""

	(X,x_names,Y,y_name,
	 IDs,IDs_name,timevar,time_name,
	 W,heteroscedasticity_factors,has_intercept,
	 mp,descr,args_archive, args,user_constraints)=setvars(loadargs,dataframe,model_string,
	                                      IDs_name,heteroscedasticity_factors,add_intercept,time_name,descr,user_constraints)
	if loadargs==2:
		settings.pqdkm.value=args_archive.pqdkm

	results_obj=results(p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name, time_name,
	                                            group_fixed_random_eff, time_fixed_eff,W,heteroscedasticity_factors,descr,dataframe,h,has_intercept,
	                                            args_archive,user_constraints,args,mp,window,loadargs)
	return results_obj
	
def setvars(loadargs,dataframe,model_string,IDs_name,heteroscedasticity_factors,add_intercept,time_name,descr,user_constraints):
	t=type(user_constraints)
	if t!=list and t!=tuple:
		print("Warning: user user_constraints must be a list of tuples. user_constraints are not applied.")	
		
	if type(model_string)==str:
		model_string=[model_string]	
	(X,x_names,Y,y_name,
	 IDs,IDs_name,timevar,time_name,
	 W,heteroscedasticity_factors,has_intercept)=model_parser.get_variables(dataframe,model_string,IDs_name,heteroscedasticity_factors,add_intercept,time_name)

	mp=mp_check(X)

	if descr==None:
		n=int(100/len(model_string))
		descr='|'.join([i[:n] for i in model_string])
	args_archive=tempstore.args_archive(descr, loadargs)

	args=args_archive.args
	
	return (X,x_names,Y,y_name,IDs,
	        IDs_name,timevar,time_name,
	        W,heteroscedasticity_factors,has_intercept,
	        mp,descr,args_archive,args,user_constraints)
	
	
class results:
	def __init__(self,p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name,time_name,
		                     group_fixed_random_eff, time_fixed_eff, W, heteroscedasticity_factors, descr, dataframe, h, has_intercept,
		                     args_archive,user_constraints,
		                     args,mp,window,loadargs):
		print ("Creating panel")
		pnl=panel.panel(p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name,group_fixed_random_eff, time_fixed_eff,W,
			            heteroscedasticity_factors,descr,dataframe,h,has_intercept,user_constraints,args,loadargs)
		
		direction=drctn.direction(pnl)
		if not mp is None:
			mp.send_dict({'panel':pnl,'direction':direction})
	
		ll,g,G,H, conv,pr,constraints,dx_norm=maximize.maximize(pnl,direction,mp,
		                        args_archive,pnl.args.args_init,True,
		                        user_constraints,window)	

		self.outputstring=pr
		self.dx_norm=dx_norm
		self.constraints=constraints
		self.ll=ll
		self.gradient=g
		self.gradient_matrix=G
		self.hessian=H
		self.converged=conv
		self.constraints=direction.constr
		self.panel=pnl


def mp_check(X):
	N,k=X[0].shape
	N=N*len(X)
	mp=None
	if ((N*(k**0.5)>200000 and os.cpu_count()>=2) or os.cpu_count()>=24) or True:#numpy all ready have multiprocessing, so there is no purpose unless you have a lot of processors or the dataset is very big
		modules='import calculus_functions as cf'
		mp=mc.multiprocess(4,modules)
		
	return mp
	
	
