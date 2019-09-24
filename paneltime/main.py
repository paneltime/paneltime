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


warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, IDs_name=None, time_name=None,
            descr=None,
            group_fixed_random_eff=2, time_fixed_eff=True, w_names=None, loadargs=1,add_intercept=True,
            h=None,user_constraints=None,window=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""

	(X,x_names,Y,y_name,
	 IDs,IDs_name,timevar,time_name,
	 W,w_names,has_intercept,
	 mp,descr,args_archive, args,user_constraints)=setvars(loadargs,dataframe,model_string,
	                                      IDs_name,w_names,add_intercept,time_name,descr,user_constraints)
	if loadargs==2:
		p,q,m,k,d=args_archive.arimagarch

	results_obj=results(p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name, time_name,
	                                            group_fixed_random_eff, time_fixed_eff,W,w_names,descr,dataframe,h,has_intercept,
	                                            args_archive,model_string,user_constraints,args,mp,window,loadargs)
	return results_obj
	
def setvars(loadargs,dataframe,model_string,IDs_name,w_names,add_intercept,time_name,descr,user_constraints):
	t=type(user_constraints)
	if t!=list and t!=tuple:
		print("Warning: user user_constraints must be a list of tuples. user_constraints are not applied.")	
		
	
	(X,x_names,Y,y_name,
	 IDs,IDs_name,timevar,time_name,
	 W,w_names,has_intercept)=model_parser.get_variables(dataframe,model_string,IDs_name,w_names,add_intercept,time_name)

	mp=mp_check(X)
	if descr==None:
		descr=model_string[:50]
	args_archive=tempstore.args_archive(descr, loadargs)

	args=args_archive.args
	
	return (X,x_names,Y,y_name,IDs,
	        IDs_name,timevar,time_name,
	        W,w_names,has_intercept,
	        mp,descr,args_archive,args,user_constraints)
	
	
class results:
	def __init__(self,p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name,time_name,
		                     group_fixed_random_eff, time_fixed_eff, W, w_names, descr, dataframe, h, has_intercept,
		                     args_archive,model_string,user_constraints,
		                     args,mp,window,loadargs):
		print ("Creating panel")
		pnl=panel.panel(p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name,group_fixed_random_eff, time_fixed_eff,W,
			            w_names,descr,dataframe,h,has_intercept,model_string,args,loadargs,user_constraints)
		
		direction=drctn.direction(pnl)
		if not mp is None:
			mp.send_dict({'panel':pnl,'direction':direction},'static dictionary')
	
		ll,g,G,H, conv,pr,constraints,dx_norm=maximize.maximize(pnl,direction,mp,
		                        args_archive,pnl.args.args_init,True,window)	

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

	
def autofit(dataframe, model_string, d=0,process_sign_level=0.05, IDs_name=None, time_name=None,
            descr=None,
            group_fixed_random_eff=2, time_fixed_eff=True, w_names=None, loadargs=True,add_intercept=True,
            h=None,user_constraints=None,window=None
            ):
	"""Same as execute, except iterates over ARIMA and GARCH coefficients to find best match"""
	
	(X,x_names,Y,y_name,
	 IDs,IDs_name,timevar,time_name,
	 W,w_names,has_intercept,
	 mp,descr,args_archive, args,user_constraints)=setvars(loadargs,dataframe,model_string,
	                                      IDs_name,w_names,add_intercept,time_name,descr,user_constraints)
	p,q,m,k=(1,1,1,1)
	if loadargs:
		p,q,m,k,dtmp=args_archive.arimagarch
		if dtmp!=d:
			print("difference argument d changed, cannot load arguments")
			args=None
		
	p_lim,q_lim,m_lim,k_lim=False,False,False,False

	while True:
		results_obj=results(p, d, q, m, k, X, Y, IDs,timevar,x_names,y_name,IDs_name,time_name,
	                                            group_fixed_random_eff, time_fixed_eff,W,w_names,descr,dataframe,h,has_intercept,
	                                            args_archive,model_string,user_constraints,
		                                        args,mp,window,loadargs)
		panel=results_obj.panel
		constraints=results_obj.constraints
		args=results_obj.ll.args_d
		diag=output.statistics(results_obj,3,simple_statistics=True,printout=False)	
		#Testing whether the highest order of each category is significant. If it is not, it is assumed
		#the maximum order for the category is found, and the order is reduced by one.  When the maximum order
		#is found for all categories, the loop ends
		p,p_lim=model_parser.check_sign(panel,diag.tsign,'rho',		p_lim,process_sign_level)
		q,q_lim=model_parser.check_sign(panel,diag.tsign,'lambda',	q_lim,process_sign_level)
		m,m_lim=model_parser.check_sign(panel,diag.tsign,'psi',		m_lim,process_sign_level,1)
		k,k_lim=model_parser.check_sign(panel,diag.tsign,'gamma',	k_lim,process_sign_level,1)
		if p_lim and q_lim and m_lim and k_lim:
			break
		a_lim,a_incr=[],[]
		for i in ([p_lim,'rho',p],[q_lim,'lambda',q],[m_lim,'psi',m],[k_lim,'gamma',k]):
			if i[0]:
				a_lim.append(i[1]+'(%s)' %(i[2]))
			else:
				a_incr.append(i[1]+'(%s)' %(i[2]))
		if len(a_lim)>0:
			print("Found maximum lag lenght for: %s" %(",".join(a_lim)))
		if len(a_incr)>0:
			print("Extending lags for: %s" %(",".join(a_incr)))
	return results_obj


def mp_check(X):
	N,k=X.shape
	mp=None
	if ((N*(k**0.5)>200000 and os.cpu_count()>=2) or os.cpu_count()>=24) or True:#numpy all ready have multiprocessing, so there is no purpose unless you have a lot of processors or the dataset is very big
		modules='import calculus_functions as cf'
		mp=mc.multiprocess(4,modules)
		
	return mp
	
	
