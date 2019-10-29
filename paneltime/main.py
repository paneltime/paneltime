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


def execute(dataframe, model_string, IDs_name, time_name, descr,settings,window):

	"""optimizes LL using the optimization procedure in the maximize module"""

	datainput=input_processor(dataframe,model_string,IDs_name,time_name, settings,descr)
	if settings.autofit:
		return autofit(dataframe, model_string,settings,window,datainput)
	if settings.loadargs==2:
		p,q,m,k,d=datainput.args_archive.arimagarch
	mp=mp_check(datainput.X)	
	results_obj=results(dataframe,datainput,settings,mp,window)
	return results_obj

class input_processor:
	def __init__(self,dataframe,model_string,IDs_name,time_name, settings,descr):
		
		t=type(settings.user_constraints)
		if t!=list and t!=tuple and (not t is None):
			print("Warning: user user_constraints must be a list of tuples. user_constraints are not applied.")	
			
		
		model_parser.get_variables(self,dataframe,model_string,IDs_name,time_name,settings)
		self.descr=descr
		if descr==None:
			self.descr=model_string[:50]
		self.args_archive=tempstore.args_archive(descr, settings.loadargs)
		self.args=self.args_archive.args

	
	
class results:
	def __init__(self,dataframe,datainput,settings,mp,window):
		print ("Creating panel")
		pnl=panel.panel(dataframe,datainput,settings)
		direction=drctn.direction(pnl)	
		if not mp is None:
			mp.send_dict({'panel':pnl,'direction':direction},'static dictionary')			
		
		ll,g,G,H, conv,pr,constraints,dx_norm=maximize.maximize(pnl,direction,mp,pnl.args.args_init,True,window)	
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

	
def autofit(dataframe, model_string,settings,window,datainput):
	"""Same as execute, except iterates over ARIMA and GARCH coefficients to find best match"""
	
	p,q,m,k=(1,1,1,1)
	if loadargs:
		p,q,dtmp,m,k=args_archive.arimagarch
		if dtmp!=d:
			print("difference argument d changed, cannot load arguments")
			args=None
	s=settings
	s.p,s.q,s.m,s.k=p,q,m,k
	p_lim,q_lim,m_lim,k_lim=False,False,False,False
	mp=mp_check(self.X)	
	while True:
		results_obj=results(dataframe,datainput,settings,mp,window)
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
	
	
