#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Todo: 



#capture singular matrix with test_small.csv
#make sure error in h function triggers an exeption


import numpy as np
import regstats
import panel
import warnings
import multi_core as mc
import loaddata
import model_parser
import maximize
import tempstore
import os
import loglikelihood as logl


warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

class results:
	def __init__(self, p, d, q, m, k, X, Y, IDs,x_names,y_name,IDs_name,
                         fixed_random_eff,W,w_names,descr,dataframe,h,has_intercept,
                         args_archive,model_string,user_constraints,
                         direction_testing,args,mp):
		ll,g,G,H, conv=maximize(panel, direction, mp, direction_testing, 
		                        args_archive,args,True,user_constraints)
		self.ll=ll
		self.g=g
		self.G=G
		self.H=H
		self.conv=conv
		self.constraints=direction.constr


def execute(dataframe, model_string, p=1, d=0, q=1, m=1, k=1, IDs_name=None, time_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=1,direction_testing=True,add_intercept=True,
            h=None,user_constraints=None
            ):

	"""optimizes LL using the optimization procedure in the maximize module"""

	if direction_testing and loadargs:
		direction_testing=False
		
	X,x_names,Y,y_name,IDs,IDs_name,W,w_names,has_intercept=model_parser.get_variables(dataframe,model_string,IDs_name,w_names,add_intercept,time_name)
	
	mp=mp_check(direction_testing, X)
		
	args_archive=tempstore.args_archive(model_string+descr, loadargs)
	if loadargs==2:
		p,q,m,k=get_args_lags(args_archive.args, loadargs)
	results_obj=results(p, d, q, m, k, X, Y, IDs,x_names,y_name,IDs_name,
	                                            fixed_random_eff,W,w_names,descr,dataframe,h,has_intercept,
	                                            args_archive,model_string,user_constraints,direction_testing,args_archive.args,mp)
	return results_obj
	
class results:
	def __init__(self,p, d, q, m, k, X, Y, IDs,x_names,y_name,IDs_name,
		                     fixed_random_eff,W,w_names,descr,dataframe,h,has_intercept,
		                     args_archive,model_string,user_constraints,
		                     direction_testing,args,mp):
		print ("Creating panel")
		pnl=panel.panel(p, d, q, m, k, X, Y, IDs,x_names,y_name,IDs_name,fixed_random_eff,W,
			            w_names,descr,dataframe,h,has_intercept,model_string,user_constraints,args)
		
		direction=logl.direction(pnl)
		if not mp is None:
			mp.send_dict({'panel':pnl,'direction':direction},'static dictionary')
			
		 
	
		print ("Maximizing:")
	
	
		ll,g,G,H, conv = maximize.maximize(pnl,direction,mp,direction_testing,args_archive,pnl.args.args,_print=True,user_constraints=user_constraints)	
		
		self.ll=ll
		self.gradient=g
		self.gradient_matrix=G
		self.hessian=H
		self.converged=conv
		self.constraints=direction.constr
		self.panel=pnl
	


def get_args_lags(args,loadargs):
	if loadargs and (not args is None):
		p,q,m,k=len(args['rho']),len(args['lambda']),len(args['gamma']),len(args['psi'])		
	else:
		p,q,m,k=1,1,1,1	
	return p,q,m,k

	
def autofit(dataframe, model_string, d=0,process_sign_level=0.05, IDs_name=None, time_name=None,
            descr="project_1",
            fixed_random_eff=2, w_names=None, loadargs=True,direction_testing=True,add_intercept=True,
            h=None,user_constraints=None
            ):
	if direction_testing and loadargs:
		direction_testing=False
		
	X,x_names,Y,y_name,IDs,IDs_name,W,w_names,has_intercept=model_parser.get_variables(dataframe,model_string,IDs_name,w_names,add_intercept,time_name)
	mp=mp_check(direction_testing, X)
		
	args_archive=tempstore.args_archive(model_string+descr, loadargs)
	
	p,q,m,k=get_args_lags(args_archive.args, loadargs)
	p_lim,q_lim,m_lim,k_lim=False,False,False,False
	args=args_archive.args
	while True:
		results_obj=results(p, d, q, m, k, X, Y, IDs,x_names,y_name,IDs_name,
	                                            fixed_random_eff,W,w_names,descr,dataframe,h,has_intercept,
	                                            args_archive,model_string,user_constraints,
		                                        direction_testing,args,mp)
		panel=results_obj.panel
		constraints=results_obj.constraints
		args=results_obj.ll.args_d
		direction_testing=False
		diag=regstats.statistics(results_obj,3,simple_diagnostics=True)	
		#Testing whether the highest order of each category is significant. If it is not, it is assumed
		#the maximum order for the category is found, and the order is reduced by one.  When the maximum order
		#is found for all categories, the loop ends
		p,p_lim=model_parser.check_sign(panel,diag.tsign,'rho',		p_lim,constraints,process_sign_level)
		q,q_lim=model_parser.check_sign(panel,diag.tsign,'lambda',	q_lim,constraints,process_sign_level)
		m,m_lim=model_parser.check_sign(panel,diag.tsign,'psi',		m_lim,constraints,process_sign_level,1)
		k,k_lim=model_parser.check_sign(panel,diag.tsign,'gamma',	k_lim,constraints,process_sign_level,1)
		loadargs=True
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


def mp_check(direction_testing,X):
	N,k=X.shape
	mp=None
	if ((N*(k**0.5)>200000 and os.cpu_count()>=2) or os.cpu_count()>=24):#numpy all ready have multiprocessing, so there is no purpose unless you have a lot of processors or the dataset is very big
		if direction_testing:
			mp=mc.multiprocess(os.cpu_count())
		else:
			mp=mc.multiprocess(4)
	return mp
	